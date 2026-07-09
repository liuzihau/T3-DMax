# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# DBet block-diffusion decoding -- the DMax analogue of generate_uniform.py's
# BlockDiffusionLLM, with the heavy per-iter forward partially REPLACED by the
# lightweight Delta-h drafter. Per block we alternate:
#   heavy forward  -> dmax_commit_uniform commits its confident LEFT prefix   [heavy]
#   drafter forward-> commits its confident LEFT-prefix extension, gated by    [draft]
#                     the TRAINED confidence head (>= draft_threshold)
# The next heavy forward re-anchors over the drafter's commits and commits more,
# until the block is fully committed. Drafter commits are TRUSTED (confidence-
# gated, not heavy-re-verified) -- that is where the speedup comes from.
#
# Reuses the commit rule + block-causal mask from generate_t3d (single source of
# truth for the heavy's decode_uniform). Plain PyTorch, single GPU; the heavy is
# the frozen DMax LLaDA2-MoE, the drafter is our trained DbetForDraftDecoding.

import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _now(device):
    """Wall clock with a CUDA sync so per-forward timing is accurate (the decode is sequential, so the sync
    adds no real overhead -- each forward already waits on the previous)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()

_HERE = os.path.dirname(os.path.abspath(__file__))                      # .../dInfer/python/dinfer/decoding
_DINFER_PYTHON = os.path.abspath(os.path.join(_HERE, "..", ".."))       # .../dInfer/python (the `dinfer` package root)
_T3DMAX_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))  # .../T3-DMax
_DFACTORY = os.path.join(_T3DMAX_ROOT, "dFactory")
for _p in (_DINFER_PYTHON, _DFACTORY, os.path.join(_DFACTORY, "VeOmni")):  # dinfer pkg + models importable
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from dinfer.decoding.generate_t3d import build_block_causal_mask, dmax_commit_uniform  # noqa: E402

MASK_ID = 156895
EOS_ID = 156892
PAD_ID = 156892


# ============================================================================
#                          model loading
# ============================================================================
def _backfill_heavy_fields(cfg, hcfg):
    """The drafter's heavy-facing base fields MUST mirror the frozen heavy (the drafter reuses its embed /
    lm_head / hiddens, and the -1 draft sizes resolve to them). Trained hf_ckpt config.jsons have shipped with
    these at CLASS DEFAULTS (hidden_size 1024, num_key_value_heads 0, ... -> ZeroDivision in DbetAttention /
    warm-start size mismatch), so mirror them from the heavy config UNCONDITIONALLY — hcfg is the ground truth
    for the heavy the drafter runs against. Also repair draft_* overrides saved as 0 (never valid; 0 means the
    field was lost -> restore -1 = match heavy)."""
    fields = ("num_key_value_heads", "num_attention_heads", "hidden_size", "intermediate_size",
              "head_dim", "rope_theta", "rms_norm_eps", "vocab_size", "max_position_embeddings")
    fixed = []
    for k in fields:
        hv = getattr(hcfg, k, None)
        if hv is not None and getattr(cfg, k, None) != hv:
            setattr(cfg, k, hv)
            fixed.append(f"{k}={hv}")
    for k in ("draft_hidden_size", "draft_num_attention_heads", "draft_num_key_value_heads",
              "draft_intermediate_size", "fuse_hidden_size", "head_intermediate_size"):
        if getattr(cfg, k, -1) == 0:
            setattr(cfg, k, -1)
            fixed.append(f"{k}: 0->-1")
    if fixed:
        print(f"[dbet] config mirrored from heavy (drafter config.json had defaults/zeros): {', '.join(fixed)}")


def load_dbet_model(drafter_path, heavy_path, device="cuda"):
    """Assemble DBet for inference: the FROZEN DMax heavy (fused MoE) + the trained drafter weights.
    `drafter_path` = the drafter-only hf_ckpt (heavy.* dropped at save; loaded strict=False).
    `heavy_path`   = the DMax-Math-16B-moe-merge checkpoint (provides heavy + embed/lm_head/final-norm)."""
    from models.dbet.configuration_dbet import DbetConfig
    from models.dbet.modeling_dbet import DbetForDraftDecoding
    from models.llada2_moe.configuration_llada2_moe import LLaDA2MoeConfig
    from models.llada2_moe.modeling_llada2_moe import LLaDA2MoeModelLM

    drafter_path = os.path.abspath(drafter_path)
    heavy_path = os.path.abspath(heavy_path)
    cfg = DbetConfig.from_pretrained(drafter_path)

    # heavy: force fused-MoE layout so the merged DMax checkpoint loads its experts (mirrors build_dbet_init)
    hcfg = LLaDA2MoeConfig.from_pretrained(heavy_path, trust_remote_code=True)
    if not str(hcfg.model_type).endswith("_veomni"):
        hcfg.model_type = str(hcfg.model_type) + "_veomni"
    hcfg.moe_implementation = "fused"
    _backfill_heavy_fields(cfg, hcfg)
    heavy = LLaDA2MoeModelLM.from_pretrained(
        heavy_path, config=hcfg, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa")

    model = DbetForDraftDecoding(cfg, _heavy=heavy)
    sd = _load_drafter_state_dict(drafter_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    drafter_missing = [k for k in missing if k.startswith("draft.") and "frozen_" not in k]
    if drafter_missing:
        print(f"[dbet] WARNING: {len(drafter_missing)} drafter params missing from ckpt (untrained?): "
              f"{drafter_missing[:4]}...")
    print(f"[dbet] loaded drafter ({len(sd)} tensors); heavy fused; sel_layers={cfg.sel_layers_list}")
    model.eval().to(device=device, dtype=torch.bfloat16)
    return model


@torch.no_grad()
def load_drafter_standalone(drafter_path, heavy_path, device="cuda", dtype=torch.bfloat16):
    """G2 (SGLang) LEAN loader: the trained DBet DRAFTER ONLY — no 16B eager heavy, no veomni/fused-MoE. Builds a
    `DbetDraftStack` and gives it the heavy's frozen embed / lm_head / final-norm loaded STANDALONE from
    `heavy_path` (~1GB: word_embeddings + lm_head + model.norm). For running the eager drafter alongside the
    SGLang heavy. Returns the DbetDraftStack in eval() on `device`/`dtype`.
    `heavy_path` MUST be the same DMax-Math checkpoint the drafter was trained with (its exact embed/lm_head/norm)."""
    import glob
    import torch.nn as nn
    from safetensors import safe_open
    from models.dbet.configuration_dbet import DbetConfig
    from models.dbet.modeling_dbet import DbetDraftStack
    from models.llada2_moe.configuration_llada2_moe import LLaDA2MoeConfig
    from models.llada2_moe.modeling_llada2_moe import LLaDA2MoeRMSNorm

    drafter_path = os.path.abspath(drafter_path); heavy_path = os.path.abspath(heavy_path)
    cfg = DbetConfig.from_pretrained(drafter_path)
    hcfg = LLaDA2MoeConfig.from_pretrained(heavy_path, trust_remote_code=True)
    _backfill_heavy_fields(cfg, hcfg)
    V, Dh, eps = hcfg.vocab_size, hcfg.hidden_size, hcfg.rms_norm_eps

    # pull ONLY embed / lm_head / final-norm from the heavy shards (safe_open = lazy, loads just these tensors;
    # model.norm is in the LAST shard so we may scan all, but only the 3 needed tensors are materialized)
    want = {"model.word_embeddings.weight", "lm_head.weight", "model.norm.weight"}
    got = {}
    for f in sorted(glob.glob(os.path.join(heavy_path, "*.safetensors"))):
        with safe_open(f, framework="pt") as sf:
            keys = set(sf.keys())
            for k in list(want):
                if k in keys:
                    got[k] = sf.get_tensor(k); want.discard(k)
        if not want:
            break
    if "model.word_embeddings.weight" not in got or "model.norm.weight" not in got:
        raise KeyError(f"{heavy_path}: missing embed/norm weights (found {list(got)})")
    emb_w = got["model.word_embeddings.weight"]
    lm_w = got.get("lm_head.weight", emb_w)                          # tied lm_head -> reuse the embedding weight

    embed = nn.Embedding(V, Dh); embed.weight = nn.Parameter(emb_w.clone(), requires_grad=False)
    lm_head = nn.Linear(Dh, V, bias=False); lm_head.weight = nn.Parameter(lm_w.clone(), requires_grad=False)
    final_norm = LLaDA2MoeRMSNorm(Dh, eps=eps)
    final_norm.weight = nn.Parameter(got["model.norm.weight"].clone(), requires_grad=False)

    draft = DbetDraftStack(cfg, embed, lm_head, final_norm)
    sd = _load_drafter_state_dict(drafter_path)
    sd = {k[len("draft."):]: v for k, v in sd.items() if k.startswith("draft.")}   # strip the ForDraftDecoding prefix
    missing, unexpected = draft.load_state_dict(sd, strict=False)
    drafter_missing = [k for k in missing if "frozen_" not in k]
    if drafter_missing:
        print(f"[drafter] WARNING: {len(drafter_missing)} params missing (untrained?): {drafter_missing[:4]}")
    if unexpected:
        print(f"[drafter] WARNING: {len(unexpected)} unexpected keys: {unexpected[:4]}")
    # frozen embed/lm_head/final-norm are PLAIN attrs (object.__setattr__) -> draft.to() won't move them; do it here
    for mod in (embed, lm_head, final_norm):
        mod.to(device=device, dtype=dtype)
    draft.eval().to(device=device, dtype=dtype)
    print(f"[drafter] standalone: {len(sd)} tensors; embed/lm_head/norm from heavy; sel_layers={cfg.sel_layers_list}; "
          f"{sum(p.numel() for p in draft.parameters()) / 1e6:.0f}M drafter params")
    return draft


def _load_drafter_state_dict(path):
    """Load the drafter safetensors (single file or sharded) into one dict."""
    from safetensors.torch import load_file
    import glob
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {path}")
    sd = {}
    for f in files:
        sd.update(load_file(f))
    return sd


# ============================================================================
#                          drafter commit (confidence-gated)
# ============================================================================
def draft_commit_confident(draft_logits, draft_conf, threshold):
    """Left-to-right prefix commit over the CANVAS (the masked block positions, in order). Commit argmax tokens
    while the trained confidence head >= threshold; STOP at the first below-threshold position (decode_uniform
    shape, but gated by the conf head instead of the logit prob). Always commits at least the leftmost canvas
    token (guarantees progress).
    draft_logits [1,C,V], draft_conf [1,C] -> (tokens [C], commit_prefix_mask [C] bool)."""
    tokens = draft_logits[0].argmax(dim=-1)                              # [C]
    conf = draft_conf[0]                                                 # [C]
    ok = conf >= threshold                                              # [C] bool
    # left-to-right run: commit positions 0..t-1 where all are ok; stop at first not-ok
    not_ok = (~ok).long()
    failed_before = torch.cumsum(not_ok, dim=0) > 0                     # True from the first below-thr onward
    commit = ~failed_before
    commit[0] = True                                                    # always commit the leftmost (progress)
    return tokens, commit


# ============================================================================
#                          soft-embedding (DMax decode_uniform input feed)
# ============================================================================
def _soft_embed(logits_sel, embed_layer, mask_id, tau, top_k):
    """DMax soft-embedding for committed positions: softmax(logits/tau) -> top-k weighted token embeds +
    residual*embed(MASK), renormalized to the expected norm. logits_sel [n,V] -> [n,D]. tau=1,k=1 == top-1
    (top1_prob*embed(top1) + (1-p)*embed(MASK), renorm). Mirrors generate_t3d.build_inputs_embeds / DMax."""
    device = logits_sel.device
    probs = torch.softmax(logits_sel.float() / max(float(tau), 1e-6), dim=-1)
    topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)                  # [n,k]
    residual = (1.0 - topk_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)   # [n,1]
    topk_emb = embed_layer(topk_idx).float()                                # [n,k,D]
    mask_emb = embed_layer(torch.tensor([mask_id], device=device)).float()  # [1,D]
    soft = (topk_emb * topk_probs.unsqueeze(-1)).sum(dim=1) + mask_emb * residual          # [n,D]
    tgt = (topk_emb.norm(dim=-1) * topk_probs).sum(dim=-1, keepdim=True) + mask_emb.norm() * residual
    soft = soft * (tgt / (soft.norm(dim=-1, keepdim=True) + 1e-6))
    return soft.to(embed_layer.weight.dtype)


# ============================================================================
#                          decode
# ============================================================================
@dataclass
class DbetGenerateStats:
    """Forward accounting split by model, to quantify the compute saving vs pure-heavy decode."""
    heavy_forwards: int = 0
    draft_forwards: int = 0
    draft_commits: int = 0      # tokens committed by the drafter (the speculative wins)
    draft_fixes: int = 0        # committed tokens the drafter OVERRODE (2nd-pass fix)
    heavy_commits: int = 0      # tokens committed by the heavy
    heavy_time: float = 0.0     # wall seconds in heavy forwards
    draft_time: float = 0.0     # wall seconds in drafter forwards
    wall_time: float = 0.0      # total decode wall seconds (this sequence)


@torch.no_grad()
def decode_block_dbet(model, x, bs, be, attn, heavy_threshold, draft_threshold,
                      max_iters, max_draft_iters, tau, stats, use_draft=True,
                      heavy_tau=1.0, heavy_top_k=1, draft_tau=1.0, draft_top_k=1,
                      draft_committed_soft=False, draft_fix=True, draft_fix_threshold=None,
                      dmax_faithful=True):
    """Decode one block, DMax-faithful. The loop = DMax exactly (heavy forward -> decode_uniform commit ->
    soft-embed re-feed -> DMax exit rule); a HEAVY forward is always first, last, and the SOLE arbiter of "done".
    The draft is inserted only BETWEEN heavy forwards as a helper: after a heavy pass that isn't done, one draft
    forward proposes over the WHOLE current block and (a) EXTENDS (commits confident still-masked slots) and
    (b) FIXES (overrides a committed slot only if its conf-head >= draft_threshold AND it disagrees).
    Block-based split (matches the heavy's block structure): the draft's PREFIX = earlier blocks [0,bs), its
    CANVAS = the whole current block [bs,be) (committed re-predictable + masked). Committed slots are shown to
    the draft as HARD tokens (default; consistent with training) or SOFT `MASK + heavy-soft-embed`
    (`draft_committed_soft`). `use_draft=False` -> byte-for-byte DMax. attn = block-causal over [0,be).
    `dmax_faithful` (default True, 2026-07-10): apply DMax's decode_uniform semantics like decode_block_heavy —
    (1) COMMITTED RE-DECODE each iter (heavy self-corrects its commits AND the drafter's), (2) EXIT on the DMax
    breakflag (all active max_prob >= 0.9 OR nothing changed) instead of "no mask left" — the block runs DMax's
    post-fill verification rounds. False = the pre-0709 LOOSE variant (exit at no-mask, commits never revised):
    fewer heavy forwards but wrong commits are unrepairable except by drafter FIX (85.1% vs 90.2% at h0.5)."""
    if draft_fix_threshold is None:                       # FIX gets its own gate; None = old single-knob behavior
        draft_fix_threshold = draft_threshold
    device = x.device
    embed = model.draft.frozen_embed
    active = (x[0:1, bs:be] == MASK_ID)                                # original decode region (all-mask at block start)
    prefix_embeds = embed(x[:, :bs])                                   # [1, bs, D] prompt + earlier blocks
    block_embeds = embed(x[:, bs:be]).clone()                         # [1, blk, D] (mask -> embed(MASK))
    block_logits = None
    # block-based prefix/canvas split for the draft: prefix=[0,bs), canvas=[bs,be) (contiguous -> gather valid)
    prefix_idx = torch.zeros(1, be, dtype=torch.bool, device=device); prefix_idx[0, :bs] = True
    canvas_idx = torch.zeros(1, be, dtype=torch.bool, device=device); canvas_idx[0, bs:be] = True

    it = 0
    while it < max_iters:
        if not dmax_faithful and not bool((x[0:1, bs:be] == MASK_ID).any()):
            break                                                     # LOOSE: stop as soon as the block is full
        inputs_embeds = torch.cat([prefix_embeds, block_embeds], dim=1)   # [1, be, D] soft feed
        # ================= HEAVY forward + DMax decode_uniform commit (unchanged from DMax) =================
        _t = _now(device)
        if use_draft:
            signals = model.extract_heavy_signals(x[:, :be], attention_mask=attn, inputs_embeds=inputs_embeds)
            block_logits = signals["logits"][:, bs:be]                # [1, blk, V]
        else:
            out = model.heavy_forward(inputs_embeds=inputs_embeds, attention_mask=attn, output_hidden_states=False)
            block_logits = out.logits[:, bs:be]
        stats.heavy_time += _now(device) - _t
        stats.heavy_forwards += 1

        curr = x[0, bs:be].clone()                                    # pre-update block tokens
        mask_idx = (curr == MASK_ID).unsqueeze(0)
        x0, high_conf_idx, max_probs, breakflag = dmax_commit_uniform(block_logits, mask_idx, active, heavy_threshold)
        if dmax_faithful:
            # DMax decode_uniform: overwrite (newly-high-conf masked | ALL committed — incl. drafter commits,
            # which the heavy may thus revise/reject) with the fresh argmax; breakflag adds the nothing-changed
            # condition so post-fill verification rounds terminate exactly like the sglang/dinfer path.
            ui_mask = high_conf_idx[0] | (active[0] & (curr != MASK_ID))
            ui = ui_mask.nonzero(as_tuple=True)[0]
            changed_any = bool((x0[0][ui] != curr[ui]).any()) if ui.numel() > 0 else False
            if ui.numel() > 0:
                x[0, bs + ui] = x0[0][ui]
            stats.heavy_commits += int(high_conf_idx[0].sum())
            breakflag = bool((max_probs[0][active[0]] >= 0.9).all()) or (not changed_any)
        else:
            hci = high_conf_idx[0].nonzero(as_tuple=True)[0]
            if hci.numel() > 0:
                x[0, bs + hci] = x0[0, hci]
                stats.heavy_commits += int(hci.numel())
        # re-soft-embed ALL committed from the heavy's fresh logits (heavy re-verifies / re-encodes)
        committed = active[0] & (x[0, bs:be] != MASK_ID)
        ci = committed.nonzero(as_tuple=True)[0]
        if ci.numel() > 0:
            block_embeds[0, ci] = _soft_embed(block_logits[0, ci], embed, MASK_ID, heavy_tau, heavy_top_k)

        # ---- DMax EXIT rule (heavy is the arbiter): block DONE -> no draft this step ----
        if dmax_faithful:
            if bool(breakflag):
                break
        elif bool(breakflag) or not bool((x[0:1, bs:be] == MASK_ID).any()):
            break
        if not use_draft:                                             # HEAVY-ONLY = pure DMax
            it += 1
            continue

        # ================= DRAFT forward (helper, between heavy passes) =================
        block_x = x[0, bs:be]                                         # state entering the draft step
        mask_pos = (block_x == MASK_ID)                              # still-masked slots (EXTEND targets)
        committed_before = active[0] & (~mask_pos)                   # committed slots (FIX candidates)
        draft_ids = x[:, :be].clone()
        if draft_committed_soft and bool(committed_before.any()):    # show committed as MASK -> E(MASK) (+ heavy soft-embed)
            _cb = torch.zeros(be, dtype=torch.bool, device=device); _cb[bs:be] = committed_before
            draft_ids[0, _cb] = MASK_ID                              # single advanced-index assign (in-place, safe)
        signals["input_ids"] = draft_ids
        signals["prefix_idx"], signals["canvas_idx"] = prefix_idx, canvas_idx
        _t = _now(device)
        d = model.draft_forward(signals, attention_mask=None, tau=tau)
        stats.draft_time += _now(device) - _t
        stats.draft_forwards += 1
        dlogits, dconf = d["logits"], d["conf"]                       # [1, blk, V], [1, blk] over the WHOLE block
        if dconf is None:                                            # no conf head -> can't gate; back to heavy
            it += 1
            continue
        darg = dlogits[0].argmax(-1)                                 # [blk]
        dc = dconf[0]                                                # [blk]

        # ---- EXTEND: left-to-right prefix commit of masked slots while conf >= draft_threshold (always >=1 for progress)
        mloc = mask_pos.nonzero(as_tuple=True)[0]                    # block-local masked indices (ascending)
        if mloc.numel() > 0:
            ok = dc[mloc] >= draft_threshold
            keep = ~(torch.cumsum((~ok).long(), 0) > 0)             # prefix up to first below-threshold
            keep[0] = True
            sel = mloc[keep]
            x[0, bs + sel] = darg[sel]
            block_embeds[0, sel] = _soft_embed(dlogits[0][sel], embed, MASK_ID, draft_tau, draft_top_k)
            stats.draft_commits += int(sel.numel())
        # ---- FIX: override a committed slot only if conf-head >= its OWN threshold AND the draft disagrees ----
        if draft_fix and bool(committed_before.any()):
            fix = committed_before & (dc >= draft_fix_threshold) & (darg != block_x)
            floc = fix.nonzero(as_tuple=True)[0]
            if floc.numel() > 0:
                x[0, bs + floc] = darg[floc]
                block_embeds[0, floc] = _soft_embed(dlogits[0][floc], embed, MASK_ID, draft_tau, draft_top_k)
                stats.draft_fixes += int(floc.numel())
        it += 1

    # safety: never leave a [MASK] in the output
    still = (x[0:1, bs:be] == MASK_ID)
    if still.any() and block_logits is not None:
        sp = still[0].nonzero(as_tuple=True)[0]
        x[0, bs + sp] = block_logits[0, sp].argmax(dim=-1)


def _new_dynamic_cache():
    """DynamicCache, with a compat shim: the heavy's forward calls `get_usable_length` (removed in newer
    transformers); for an unbounded DynamicCache that equals `get_seq_length`, so alias it if missing."""
    from transformers import DynamicCache
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, new_seq_length=0, layer_idx=0: self.get_seq_length(layer_idx)
    return DynamicCache()


def _crop_cache(cache, length):
    """Truncate a DynamicCache to `length` tokens (drop the in-progress block KV). Uses `.crop` if present,
    else truncates key/value tensors directly (version-robust)."""
    if hasattr(cache, "crop"):
        cache.crop(length); return
    for i in range(len(cache.key_cache)):
        cache.key_cache[i] = cache.key_cache[i][..., :length, :].contiguous()
        cache.value_cache[i] = cache.value_cache[i][..., :length, :].contiguous()
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = length


@torch.no_grad()
def decode_block_dbet_cached(model, x, bs, be, settled, heavy_cache, draft_cache, block_length,
                             heavy_threshold, draft_threshold, max_iters, tau, stats,
                             heavy_tau=1.0, heavy_top_k=1, draft_tau=1.0, draft_top_k=1,
                             draft_committed_soft=False, draft_fix=True, draft_fix_threshold=None):
    """INCREMENTAL cross-block prefix-KV cache (heavy AND draft). Persistent caches hold the settled prefix
    [0, settled); on entry `settled == bs - blk` (or 0 for the first block) and the caches hold [0, settled).
    Returns (heavy_cache, draft_cache, new_settled=bs).

    ITER 0 = COMBINED forward (the user's design; goes BEYOND DMax, which re-forwards the whole prefix each
    block): forward the newly-SETTLED region [settled, bs) as HARD embeds (mirrors the no-cache path feeding
    completed blocks as embed(x)) TOGETHER with the current block [bs,be), reusing the cached prefix KV
    [0,settled). One pass thus (a) FINALIZES the previous block's KV -> appended to the heavy cache (no separate
    finalization forward), (b) yields that region's settled h_sel -> EXTENDS the draft prefix cache, and
    (c) yields the current block's iter-0 logits/h_sel/h_last. Then crop the heavy cache back to bs.
    ITERS 1+ = crop to bs, partial-forward only [bs,be) reusing the settled prefix KV.
    The draft reads its cached prefix K/V (constant within the block), so its k_pre/v_pre are derived once per
    settled block, not re-fused every round. Correct = iso with decode_block_dbet up to bf16 non-associativity
    (the prefix is settled-HARD + block-causally isolated, so its cached KV/h_sel equal a fresh HARD forward's)."""
    if draft_fix_threshold is None:                       # FIX gets its own gate; None = old single-knob behavior
        draft_fix_threshold = draft_threshold
    device = x.device
    embed = model.draft.frozen_embed
    mdt = embed.weight.dtype
    active = (x[0:1, bs:be] == MASK_ID)                                  # decode region (excludes prompt tail)
    finalize_len = bs - settled                                         # region [settled,bs): prompt (b0) or prev block
    m_full = build_block_causal_mask(be, block_length, dtype=mdt, device=device)   # block-causal over [0,be)
    block_embeds = embed(x[:, bs:be]).clone()                          # [1, blk, D] current block (mask embeds)
    block_hsel = block_hlast = block_logits = None

    it = 0
    while it < max_iters and bool((x[0:1, bs:be] == MASK_ID).any()):
        _t = _now(device)
        if it == 0:
            # COMBINED forward: [ HARD settled region [settled,bs) ; current block [bs,be) ] over cached prefix
            _crop_cache(heavy_cache, settled)                          # ensure the cache is exactly [0,settled)
            hard_prefix = embed(x[:, settled:bs])                      # [1, finalize_len, D] HARD (== no-cache prefix feed)
            inputs_embeds = torch.cat([hard_prefix, block_embeds], dim=1) if finalize_len > 0 else block_embeds
            m = m_full[:, :, settled:be, :]                            # queries [settled,be) attend keys [0,be) block-causally
            sig = model.extract_heavy_signals(x[:, settled:be], attention_mask=m, inputs_embeds=inputs_embeds,
                                              past_key_values=heavy_cache, use_cache=True)
            heavy_cache = sig["past_key_values"]                       # now [0,be)
            _crop_cache(heavy_cache, bs)                               # finalize prev block, drop current-block provisional KV
            if finalize_len > 0:                                       # EXTEND the draft prefix cache by the settled region
                pos = torch.arange(settled, bs, device=device).unsqueeze(0)
                model.extend_draft_prefix_cache(sig["h_sel"][:, :finalize_len], pos, draft_cache)
            block_logits = sig["logits"][:, finalize_len:]
            block_hsel, block_hlast = sig["h_sel"][:, finalize_len:], sig["h_last"][:, finalize_len:]
        else:
            # ITERS 1+: crop the block KV off, partial-forward [bs,be) reusing the settled prefix
            _crop_cache(heavy_cache, bs)
            m = m_full[:, :, bs:be, :]
            sig = model.extract_heavy_signals(x[:, bs:be], attention_mask=m, inputs_embeds=block_embeds,
                                              past_key_values=heavy_cache, use_cache=True)
            heavy_cache = sig["past_key_values"]                       # [0,be)
            block_logits, block_hsel, block_hlast = sig["logits"], sig["h_sel"], sig["h_last"]
        stats.heavy_time += _now(device) - _t; stats.heavy_forwards += 1

        mask_idx = (x[0:1, bs:be] == MASK_ID)
        x0, high_conf_idx, _, breakflag = dmax_commit_uniform(block_logits, mask_idx, active, heavy_threshold)
        hci = high_conf_idx[0].nonzero(as_tuple=True)[0]
        if hci.numel() > 0:
            x[0, bs + hci] = x0[0, hci]; stats.heavy_commits += int(hci.numel())
        committed = active[0] & (x[0, bs:be] != MASK_ID)
        ci = committed.nonzero(as_tuple=True)[0]
        if ci.numel() > 0:
            block_embeds[0, ci] = _soft_embed(block_logits[0, ci], embed, MASK_ID, heavy_tau, heavy_top_k)
        if bool(breakflag) or not bool((x[0:1, bs:be] == MASK_ID).any()):
            break

        # ---- draft: canvas = current block, prefix K/V read from the cross-block draft cache (h_sel_prefix=None) ----
        block_x = x[0, bs:be]; mask_pos = (block_x == MASK_ID)
        committed_before = active[0] & (~mask_pos)
        draft_ids = x[:, bs:be].clone()
        if draft_committed_soft and bool(committed_before.any()):
            draft_ids[0][committed_before] = MASK_ID
        _t = _now(device)
        d = model.draft(input_ids=draft_ids, heavy_logits=block_logits, h_sel_denoise=block_hsel,
                        h_last_denoise=block_hlast, h_sel_prefix=None, past_key_values=draft_cache,
                        attention_mask=None, position_ids=None, denoise_mask=None, tau=tau)
        stats.draft_time += _now(device) - _t; stats.draft_forwards += 1
        dlogits, dconf = d["logits"], d["conf"]
        if dconf is None:
            it += 1; continue
        darg = dlogits[0].argmax(-1); dc = dconf[0]
        mloc = mask_pos.nonzero(as_tuple=True)[0]
        if mloc.numel() > 0:
            ok = dc[mloc] >= draft_threshold
            keep = ~(torch.cumsum((~ok).long(), 0) > 0); keep[0] = True
            sel = mloc[keep]
            x[0, bs + sel] = darg[sel]
            block_embeds[0, sel] = _soft_embed(dlogits[0][sel], embed, MASK_ID, draft_tau, draft_top_k)
            stats.draft_commits += int(sel.numel())
        if draft_fix and bool(committed_before.any()):
            fix = committed_before & (dc >= draft_fix_threshold) & (darg != block_x)
            floc = fix.nonzero(as_tuple=True)[0]
            if floc.numel() > 0:
                x[0, bs + floc] = darg[floc]
                block_embeds[0, floc] = _soft_embed(dlogits[0][floc], embed, MASK_ID, draft_tau, draft_top_k)
                stats.draft_fixes += int(floc.numel())
        it += 1

    # safety: never leave a [MASK] in the output
    still = (x[0:1, bs:be] == MASK_ID)
    if still.any() and block_logits is not None:
        sp = still[0].nonzero(as_tuple=True)[0]
        x[0, bs + sp] = block_logits[0, sp].argmax(dim=-1)
    _crop_cache(heavy_cache, bs)                                        # leave the cache at the settled prefix [0,bs)
    return heavy_cache, draft_cache, bs


@torch.no_grad()
def decode_block_heavy(model, x, bs, be, attn, threshold, max_iters, stats,
                       heavy_tau=1.0, heavy_top_k=1):
    """HEAVY-ONLY block decode = a FAITHFUL mirror of DMax's `ThresholdParallelDecoder.decode_uniform`
    (parallel_strategy.py:542) + block loop (generate_uniform.py:319), so heavy-only == DMax's algorithm.
    The selector (`dmax_commit_uniform` == `get_transfer_index_uniform`) and the top-1 soft-embed re-feed
    already match; this adds the two DMax behaviours the drafter-oriented `decode_block_dbet` intentionally
    drops (drafter logic there is UNTOUCHED):
      (1) COMMITTED RE-DECODE: each iter overwrites ALL committed positions with the fresh heavy argmax
          (`update = high_conf | (active & ~mask)`) -> the heavy self-corrects its own commits.
      (2) EXIT on the DMax breakflag (`all active max_prob >= 0.9` OR `nothing changed`), running the loop to
          breakflag rather than to "no mask left" -> includes DMax's post-fill stability forward.
    No drafter, no KV cache (cache is a separate, algorithm-neutral speed knob)."""
    device = x.device
    embed = model.draft.frozen_embed
    active = (x[0:1, bs:be] == MASK_ID)                          # decode region (all-mask at block start)
    prefix_embeds = embed(x[:, :bs])
    block_embeds = embed(x[:, bs:be]).clone()                   # iter input feed (== DMax prev_embeddings)
    block_logits = None

    it = 0
    while it < max_iters:
        inputs_embeds = torch.cat([prefix_embeds, block_embeds], dim=1)
        _t = _now(device)
        out = model.heavy_forward(inputs_embeds=inputs_embeds, attention_mask=attn, output_hidden_states=False)
        block_logits = out.logits[:, bs:be]
        stats.heavy_time += _now(device) - _t; stats.heavy_forwards += 1

        curr = x[0, bs:be].clone()                              # pre-update block tokens
        mask_idx = (curr == MASK_ID).unsqueeze(0)              # [1, blk]
        x0, high_conf_idx, max_probs, _ = dmax_commit_uniform(block_logits, mask_idx, active, threshold)
        x0b, hci = x0[0], high_conf_idx[0]
        # DMax: overwrite (newly-high-conf masked | already-committed) with the fresh argmax
        ui = (hci | (active[0] & (curr != MASK_ID))).nonzero(as_tuple=True)[0]
        changed_any = bool((x0b[ui] != curr[ui]).any()) if ui.numel() > 0 else False   # BEFORE overwrite
        if ui.numel() > 0:
            x[0, bs + ui] = x0b[ui]
        stats.heavy_commits += int(hci.sum())
        # DMax breakflag: all active >= 0.9 (empty active -> all()=True) OR nothing changed
        breakflag = bool((max_probs[0][active[0]] >= 0.9).all()) or (not changed_any)
        # soft-embed re-feed: committed (active & ~new_mask) -> soft-embed; still-masked -> embed(MASK)
        new_curr = x[0, bs:be]
        block_embeds = embed(x[:, bs:be]).clone()
        soft = (active[0] & (new_curr != MASK_ID)).nonzero(as_tuple=True)[0]
        if soft.numel() > 0:
            block_embeds[0, soft] = _soft_embed(block_logits[0, soft], embed, MASK_ID, heavy_tau, heavy_top_k)
        if breakflag:
            break
        it += 1

    # safety: never leave a [MASK] in the output
    still = (x[0:1, bs:be] == MASK_ID)
    if still.any() and block_logits is not None:
        sp = still[0].nonzero(as_tuple=True)[0]
        x[0, bs + sp] = block_logits[0, sp].argmax(dim=-1)


@torch.no_grad()
def generate_heavy(model, prompt_ids, gen_length, block_length,
                   heavy_threshold=0.9, max_iter_per_block=32, early_stop=True,
                   heavy_tau=1.0, heavy_top_k=1):
    """HEAVY-ONLY baseline (pure DMax block-diffusion decode_uniform through this model): no drafter, no
    hidden-state collection, soft-embedding re-feed. Same block loop + commit rule as generate_dbet, so
    `heavy_forwards` is directly comparable."""
    return generate_dbet(model, prompt_ids, gen_length, block_length,
                         heavy_threshold=heavy_threshold, max_iter_per_block=max_iter_per_block,
                         early_stop=early_stop, use_draft=False, heavy_tau=heavy_tau, heavy_top_k=heavy_top_k)


@torch.no_grad()
def generate_dbet(model, prompt_ids, gen_length, block_length,
                  heavy_threshold=0.9, draft_threshold=0.7, max_iter_per_block=32,
                  max_draft_iters=1, tau=None, early_stop=True, use_draft=True,
                  heavy_tau=1.0, heavy_top_k=1, draft_tau=1.0, draft_top_k=1,
                  draft_committed_soft=False, draft_fix=True, draft_fix_threshold=None,
                  use_cache=False, progress_desc=None, dmax_faithful=True):
    """Grid-aligned multi-block DBet generation. Returns (response_ids [n], DbetGenerateStats); response_ids
    excludes the prompt and is cut at the first EOS.
    heavy_threshold: decode_uniform commit confidence for the HEAVY (DMax default 0.9 here for high precision).
    draft_threshold: the trained confidence-head gate for committing DRAFTER tokens (higher = safer/slower).
    use_draft=False -> pure heavy-only baseline (see generate_heavy).
    progress_desc: if given (a label), show a per-block tqdm bar (useful for the very slow --exact_moe proof)."""
    device = prompt_ids.device
    P = prompt_ids.shape[1]

    first_block_start = (P // block_length) * block_length
    end_target = P + gen_length
    num_blocks = (end_target - first_block_start + block_length - 1) // block_length
    L = first_block_start + num_blocks * block_length

    x = torch.full((1, L), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids

    stats = DbetGenerateStats()
    eos_cut = L
    cached = use_cache and use_draft                                   # incremental cross-block prefix-KV cache (DBet only)
    if cached and dmax_faithful:
        print("[dbet] NOTE: --use_cache path keeps the pre-0709 LOOSE exit semantics (no committed re-decode / "
              "no post-fill verification); faithful mode applies to the no-cache path only.")
    # persistent caches spanning ALL blocks: hold the settled prefix [0, settled); grown one block at a time.
    heavy_cache = _new_dynamic_cache() if cached else None
    draft_cache = _new_dynamic_cache() if cached else None
    settled = 0                                                        # length of prefix currently in the caches
    _t_wall = _now(device)
    block_iter = range(num_blocks)
    if progress_desc is not None:
        from tqdm import tqdm
        block_iter = tqdm(block_iter, desc=progress_desc, total=num_blocks, unit="blk", leave=False)
    for b in block_iter:
        bs = first_block_start + b * block_length
        be = bs + block_length
        if cached:
            heavy_cache, draft_cache, settled = decode_block_dbet_cached(
                model, x, bs, be, settled, heavy_cache, draft_cache, block_length,
                heavy_threshold, draft_threshold, max_iter_per_block, tau, stats,
                heavy_tau=heavy_tau, heavy_top_k=heavy_top_k,
                draft_tau=draft_tau, draft_top_k=draft_top_k,
                draft_committed_soft=draft_committed_soft, draft_fix=draft_fix,
                draft_fix_threshold=draft_fix_threshold)
        else:
            attn = build_block_causal_mask(be, block_length, dtype=model.draft.frozen_embed.weight.dtype, device=device)
            if use_draft:
                decode_block_dbet(model, x, bs, be, attn, heavy_threshold, draft_threshold,
                                  max_iter_per_block, max_draft_iters, tau, stats, use_draft=True,
                                  heavy_tau=heavy_tau, heavy_top_k=heavy_top_k,
                                  draft_tau=draft_tau, draft_top_k=draft_top_k,
                                  draft_committed_soft=draft_committed_soft, draft_fix=draft_fix,
                                  draft_fix_threshold=draft_fix_threshold, dmax_faithful=dmax_faithful)
            else:                                             # heavy-only = faithful DMax mirror (not decode_block_dbet)
                decode_block_heavy(model, x, bs, be, attn, heavy_threshold, max_iter_per_block, stats,
                                   heavy_tau=heavy_tau, heavy_top_k=heavy_top_k)
        if early_stop:
            resp_lo = max(P, bs)
            seg = x[0, resp_lo:be]
            eos_pos = (seg == EOS_ID).nonzero(as_tuple=True)[0]
            if eos_pos.numel() > 0:
                eos_cut = resp_lo + int(eos_pos[0].item())
                if be < L:
                    x[0, be:] = PAD_ID
                break

    stats.wall_time = _now(device) - _t_wall
    return x[0, P:eos_cut].clone(), stats


# ============================================================================
#                          __main__ smoke
# ============================================================================
def _smoke_test():
    """python -m dinfer.decoding.generate_dbet --drafter_path <hf_ckpt> --heavy_path <DMax> [--tokenizer_path]"""
    import argparse
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--drafter_path", required=True)
    p.add_argument("--heavy_path", required=True)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--prompt", default="What is 7 * 8?")
    p.add_argument("--gen_length", type=int, default=128)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--heavy_threshold", type=float, default=0.9)
    p.add_argument("--draft_threshold", type=float, default=0.7)
    p.add_argument("--max_draft_iters", type=int, default=1)
    p.add_argument("--heavy_only", action="store_true", help="pure-DMax baseline: no drafter.")
    p.add_argument("--heavy_tau", type=float, default=1.0, help="soft-embed temperature for heavy commits.")
    p.add_argument("--heavy_top_k", type=int, default=1, help="soft-embed top-k for heavy commits.")
    p.add_argument("--draft_tau", type=float, default=1.0, help="soft-embed temperature for drafter commits.")
    p.add_argument("--draft_top_k", type=int, default=1, help="soft-embed top-k for drafter commits.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    tok_path = os.path.abspath(args.tokenizer_path or args.heavy_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    model = load_dbet_model(args.drafter_path, args.heavy_path, args.device)

    messages = [{"role": "user", "content": args.prompt + "\nLet's think step by step\n"}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(args.device)
    response_ids, stats = generate_dbet(
        model, prompt_ids, gen_length=args.gen_length, block_length=args.block_length,
        heavy_threshold=args.heavy_threshold, draft_threshold=args.draft_threshold,
        max_draft_iters=args.max_draft_iters, use_draft=not args.heavy_only,
        heavy_tau=args.heavy_tau, heavy_top_k=args.heavy_top_k,
        draft_tau=args.draft_tau, draft_top_k=args.draft_top_k)
    text = tokenizer.decode(response_ids, skip_special_tokens=True)
    print(f"[dbet] mode={'HEAVY-ONLY' if args.heavy_only else 'DBet (heavy+draft)'}")
    n_tok = int(response_ids.shape[0])
    hpf = stats.heavy_time / max(stats.heavy_forwards, 1)
    dpf = stats.draft_time / max(stats.draft_forwards, 1)
    print(f"[dbet] heavy_fwd={stats.heavy_forwards} draft_fwd={stats.draft_forwards} "
          f"heavy_commits={stats.heavy_commits} draft_commits={stats.draft_commits}")
    print(f"[dbet] wall={stats.wall_time:.2f}s  heavy={stats.heavy_time:.2f}s ({hpf*1e3:.0f}ms/fwd)  "
          f"draft={stats.draft_time:.2f}s ({dpf*1e3:.0f}ms/fwd)  "
          f"tok={n_tok}  {n_tok / max(stats.wall_time, 1e-6):.1f} tok/s")
    print(f"[dbet] answer: {text!r}")


if __name__ == "__main__":
    _smoke_test()
