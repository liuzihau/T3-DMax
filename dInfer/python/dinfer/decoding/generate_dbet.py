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
                      draft_committed_soft=False, draft_fix=True):
    """Decode one block, DMax-faithful. The loop = DMax exactly (heavy forward -> decode_uniform commit ->
    soft-embed re-feed -> DMax exit rule); a HEAVY forward is always first, last, and the SOLE arbiter of "done".
    The draft is inserted only BETWEEN heavy forwards as a helper: after a heavy pass that isn't done, one draft
    forward proposes over the WHOLE current block and (a) EXTENDS (commits confident still-masked slots) and
    (b) FIXES (overrides a committed slot only if its conf-head >= draft_threshold AND it disagrees).
    Block-based split (matches the heavy's block structure): the draft's PREFIX = earlier blocks [0,bs), its
    CANVAS = the whole current block [bs,be) (committed re-predictable + masked). Committed slots are shown to
    the draft as HARD tokens (default; consistent with training) or SOFT `MASK + heavy-soft-embed`
    (`draft_committed_soft`). `use_draft=False` -> byte-for-byte DMax. attn = block-causal over [0,be)."""
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
    while it < max_iters and bool((x[0:1, bs:be] == MASK_ID).any()):
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

        mask_idx = (x[0:1, bs:be] == MASK_ID)
        x0, high_conf_idx, _, breakflag = dmax_commit_uniform(block_logits, mask_idx, active, heavy_threshold)
        hci = high_conf_idx[0].nonzero(as_tuple=True)[0]
        if hci.numel() > 0:
            x[0, bs + hci] = x0[0, hci]
            stats.heavy_commits += int(hci.numel())
        # re-soft-embed ALL committed from the heavy's fresh logits (heavy re-verifies / re-encodes)
        committed = active[0] & (x[0, bs:be] != MASK_ID)
        ci = committed.nonzero(as_tuple=True)[0]
        if ci.numel() > 0:
            block_embeds[0, ci] = _soft_embed(block_logits[0, ci], embed, MASK_ID, heavy_tau, heavy_top_k)

        # ---- DMax EXIT rule (heavy is the arbiter): breakflag OR no mask left -> block DONE, no draft this step ----
        if bool(breakflag) or not bool((x[0:1, bs:be] == MASK_ID).any()):
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
        # ---- FIX: override a committed slot only if conf-head >= draft_threshold AND the draft disagrees ----
        if draft_fix and bool(committed_before.any()):
            fix = committed_before & (dc >= draft_threshold) & (darg != block_x)
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
def _build_prefix_cache(model, x, upto, block_length, heavy_tau, heavy_top_k):
    """Build the initial prefix cache over the settled HARD region [0, upto) (prompt-prefix). Returns
    (DynamicCache with KV for [0,upto), prefix h_sel [1,upto,mD]). Block-causal mask over [0,upto)."""
    device = x.device
    if upto <= 0:
        return _new_dynamic_cache(), None
    embed = model.draft.frozen_embed
    attn = build_block_causal_mask(upto, block_length, dtype=model.draft.frozen_embed.weight.dtype, device=device)
    cache = _new_dynamic_cache()
    sig = model.extract_heavy_signals(x[:, :upto], attention_mask=attn, inputs_embeds=embed(x[:, :upto]),
                                      past_key_values=cache, use_cache=True)
    return sig["past_key_values"], sig["h_sel"]                          # cache=[0,upto), prefix h_sel [1,upto,mD]


@torch.no_grad()
def decode_block_dbet_cached(model, x, bs, be, heavy_threshold, draft_threshold,
                             max_iters, tau, stats, heavy_tau=1.0, heavy_top_k=1, draft_tau=1.0, draft_top_k=1,
                             draft_committed_soft=False, draft_fix=True):
    """DMax-style prefix KV cache, PER BLOCK (mirrors generate_cache.py: cache is fresh each block, no
    cross-block carry, no finalization). ITER 0 = FULL [0,be) forward with use_cache -- identical to the
    no-cache forward -- which builds the cache and yields the settled-prefix h_sel. ITERS 1+ = crop the block
    KV off and PARTIAL-forward only [bs,be) reusing the fixed prefix KV (crop+append == DMax replace_position).
    Correct = iso with decode_block_dbet up to bf16 non-associativity (proven: fp32 --exact_moe -> identical)."""
    device = x.device
    embed = model.draft.frozen_embed
    mdt = model.draft.frozen_embed.weight.dtype
    active = (x[0:1, bs:be] == MASK_ID)                                  # decode region (excludes prompt tail)
    blk = be - bs
    prefix_embeds = embed(x[:, :bs])                                    # [1, bs, D] settled prefix (HARD)
    block_embeds = embed(x[:, bs:be]).clone()                          # [1, blk, D]
    m_full = build_block_causal_mask(be, blk, dtype=mdt, device=device)   # iter 0: block-causal over [0,be)
    m_blk = torch.zeros(1, 1, blk, be, dtype=mdt, device=device)         # iters 1+: block attends all [0,be)
    heavy_cache = None; prefix_hsel = None; block_logits = None

    it = 0
    while it < max_iters and bool((x[0:1, bs:be] == MASK_ID).any()):
        _t = _now(device)
        if heavy_cache is None:
            # ITER 0: full [0,be) forward -> build the cache + the fixed prefix h_sel  (== the no-cache forward)
            inputs_embeds = torch.cat([prefix_embeds, block_embeds], dim=1)
            heavy_cache = _new_dynamic_cache()
            sig = model.extract_heavy_signals(x[:, :be], attention_mask=m_full, inputs_embeds=inputs_embeds,
                                              past_key_values=heavy_cache, use_cache=True)
            heavy_cache = sig["past_key_values"]                       # [0,be)
            prefix_hsel = sig["h_sel"][:, :bs]                          # settled prefix h_sel (fixed for the block)
            block_logits, block_hsel, block_hlast = sig["logits"][:, bs:be], sig["h_sel"][:, bs:be], sig["h_last"][:, bs:be]
        else:
            # ITERS 1+: crop the block KV off, partial-forward [bs,be) reusing the prefix (crop+append = replace)
            _crop_cache(heavy_cache, bs)
            sig = model.extract_heavy_signals(x[:, bs:be], attention_mask=m_blk, inputs_embeds=block_embeds,
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

        # ---- draft: direct call with the block canvas + cached prefix h_sel (PrefixFuse recomputed; Phase 2 caches it)
        block_x = x[0, bs:be]; mask_pos = (block_x == MASK_ID)
        committed_before = active[0] & (~mask_pos)
        draft_ids = x[:, bs:be].clone()
        if draft_committed_soft and bool(committed_before.any()):
            draft_ids[0][committed_before] = MASK_ID
        _t = _now(device)
        d = model.draft(input_ids=draft_ids, heavy_logits=block_logits, h_sel_denoise=block_hsel,
                        h_last_denoise=block_hlast, h_sel_prefix=prefix_hsel,
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
            fix = committed_before & (dc >= draft_threshold) & (darg != block_x)
            floc = fix.nonzero(as_tuple=True)[0]
            if floc.numel() > 0:
                x[0, bs + floc] = darg[floc]
                block_embeds[0, floc] = _soft_embed(dlogits[0][floc], embed, MASK_ID, draft_tau, draft_top_k)
                stats.draft_fixes += int(floc.numel())
        it += 1

    # safety: never leave a [MASK] in the output (no finalization -- the cache is rebuilt fresh next block)
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
                  draft_committed_soft=False, draft_fix=True, use_cache=False):
    """Grid-aligned multi-block DBet generation. Returns (response_ids [n], DbetGenerateStats); response_ids
    excludes the prompt and is cut at the first EOS.
    heavy_threshold: decode_uniform commit confidence for the HEAVY (DMax default 0.9 here for high precision).
    draft_threshold: the trained confidence-head gate for committing DRAFTER tokens (higher = safer/slower).
    use_draft=False -> pure heavy-only baseline (see generate_heavy)."""
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
    cached = use_cache and use_draft                                   # DMax-style per-block prefix-KV cache (DBet only)
    _t_wall = _now(device)
    for b in range(num_blocks):
        bs = first_block_start + b * block_length
        be = bs + block_length
        if cached:
            decode_block_dbet_cached(
                model, x, bs, be, heavy_threshold, draft_threshold,
                max_iter_per_block, tau, stats, heavy_tau=heavy_tau, heavy_top_k=heavy_top_k,
                draft_tau=draft_tau, draft_top_k=draft_top_k,
                draft_committed_soft=draft_committed_soft, draft_fix=draft_fix)
        else:
            attn = build_block_causal_mask(be, block_length, dtype=model.draft.frozen_embed.weight.dtype, device=device)
            decode_block_dbet(model, x, bs, be, attn, heavy_threshold, draft_threshold,
                              max_iter_per_block, max_draft_iters, tau, stats, use_draft=use_draft,
                              heavy_tau=heavy_tau, heavy_top_k=heavy_top_k,
                              draft_tau=draft_tau, draft_top_k=draft_top_k,
                              draft_committed_soft=draft_committed_soft, draft_fix=draft_fix)
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
