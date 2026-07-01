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
    heavy_commits: int = 0      # tokens committed by the heavy
    heavy_time: float = 0.0     # wall seconds in heavy forwards
    draft_time: float = 0.0     # wall seconds in drafter forwards
    wall_time: float = 0.0      # total decode wall seconds (this sequence)


@torch.no_grad()
def decode_block_dbet(model, x, bs, be, attn, heavy_threshold, draft_threshold,
                      max_iters, max_draft_iters, tau, stats, use_draft=True,
                      heavy_tau=1.0, heavy_top_k=1, draft_tau=1.0, draft_top_k=1):
    """Decode one block in place with DMax soft-embedding re-feed. Committed positions are fed back to the heavy
    as a SOFT embed (softmax(logits/tau) -> top-k weighted + residual mask, renormalized), tracked per
    PROVENANCE: heavy-committed use (heavy_tau, heavy_top_k) [DMax default 1.0/1 = top-1], drafter-committed use
    (draft_tau, draft_top_k). `x` holds the hard argmax ids (for the prefix/canvas split + final output);
    `block_embeds` holds the soft feed. If use_draft: alternate heavy-commit / drafter-extend; else HEAVY-ONLY
    baseline (pure DMax decode_uniform). bs/be = block start/end; attn = block-causal 4D mask over [0,be)."""
    device = x.device
    embed = model.draft.frozen_embed                                    # frozen heavy embedding
    active = (x[0:1, bs:be] == MASK_ID)                                 # original decode region
    prefix_embeds = embed(x[:, :bs])                                    # [1, bs, D] prompt + earlier blocks (hard)
    block_embeds = embed(x[:, bs:be]).clone()                          # [1, blk, D] (mask positions -> embed(MASK))
    block_logits = None

    it = 0
    while it < max_iters and bool((x[0:1, bs:be] == MASK_ID).any()):
        inputs_embeds = torch.cat([prefix_embeds, block_embeds], dim=1)  # [1, be, D] soft feed
        # ---- heavy forward (on soft embeds) + decode_uniform commit ----
        _t = _now(device)
        if use_draft:
            signals = model.extract_heavy_signals(x[:, :be], attention_mask=attn, inputs_embeds=inputs_embeds)
            block_logits = signals["logits"][:, bs:be]                 # [1, blk, V]
        else:
            out = model.heavy_forward(inputs_embeds=inputs_embeds, attention_mask=attn, output_hidden_states=False)
            block_logits = out.logits[:, bs:be]                        # heavy-only: logits only, no hidden
        stats.heavy_time += _now(device) - _t
        stats.heavy_forwards += 1

        mask_idx = (x[0:1, bs:be] == MASK_ID)
        n_before = int(mask_idx.sum())
        x0, high_conf_idx, _, _ = dmax_commit_uniform(block_logits, mask_idx, active, heavy_threshold)
        hci = high_conf_idx[0].nonzero(as_tuple=True)[0]               # block-local indices the heavy commits
        if hci.numel() > 0:
            x[0, bs + hci] = x0[0, hci]
            block_embeds[0, hci] = _soft_embed(block_logits[0, hci], embed, MASK_ID, heavy_tau, heavy_top_k)
            stats.heavy_commits += int(hci.numel())
        if not bool((x[0:1, bs:be] == MASK_ID).any()):
            break
        if not use_draft:                                              # HEAVY-ONLY: loop (soft re-feed)
            it += 1
            continue

        # ---- drafter extend (confidence-gated); drafter-committed positions get their OWN soft embed ----
        for _di in range(max_draft_iters):
            if not bool((x[0:1, bs:be] == MASK_ID).any()):
                break
            signals["input_ids"] = x[:, :be]
            signals["prefix_idx"], signals["canvas_idx"] = model._split_prefix_denoise(x[:, :be])
            _t = _now(device)
            d = model.draft_forward(signals, attention_mask=None, tau=tau)
            stats.draft_time += _now(device) - _t
            stats.draft_forwards += 1
            dlogits, dconf = d["logits"], d["conf"]                     # [1,C,V], [1,C] over canvas
            if dconf is None:                                          # no conf head -> can't gate; back to heavy
                break
            tokens, commit = draft_commit_confident(dlogits, dconf, draft_threshold)
            canvas_local = (x[0, bs:be] == MASK_ID).nonzero(as_tuple=True)[0]   # block-local masked indices (asc)
            sel = canvas_local[commit]                                 # block-local positions to commit
            if sel.numel() == 0:
                break
            x[0, bs + sel] = tokens[commit]
            block_embeds[0, sel] = _soft_embed(dlogits[0][commit], embed, MASK_ID, draft_tau, draft_top_k)
            stats.draft_commits += int(sel.numel())

        n_after = int((x[0:1, bs:be] == MASK_ID).sum())
        if n_after == n_before:                                        # neither progressed -> force one token
            mp = (x[0, bs:be] == MASK_ID).nonzero(as_tuple=True)[0]
            if mp.numel() > 0:
                p = int(mp[0])
                x[0, bs + p] = int(block_logits[0, p].argmax())
                block_embeds[0, p] = _soft_embed(block_logits[0, p:p + 1], embed, MASK_ID, heavy_tau, heavy_top_k)[0]
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
                  heavy_tau=1.0, heavy_top_k=1, draft_tau=1.0, draft_top_k=1):
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
    _t_wall = _now(device)
    for b in range(num_blocks):
        bs = first_block_start + b * block_length
        be = bs + block_length
        attn = build_block_causal_mask(be, block_length, dtype=torch.bfloat16, device=device)
        decode_block_dbet(model, x, bs, be, attn, heavy_threshold, draft_threshold,
                          max_iter_per_block, max_draft_iters, tau, stats, use_draft=use_draft,
                          heavy_tau=heavy_tau, heavy_top_k=heavy_top_k,
                          draft_tau=draft_tau, draft_top_k=draft_top_k)
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
