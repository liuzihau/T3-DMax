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
from dataclasses import dataclass

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))                      # .../dInfer/python/dinfer/decoding
_T3DMAX_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))  # .../T3-DMax
_DFACTORY = os.path.join(_T3DMAX_ROOT, "dFactory")
for _p in (_DFACTORY, os.path.join(_DFACTORY, "VeOmni")):
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
#                          decode
# ============================================================================
@dataclass
class DbetGenerateStats:
    """Forward accounting split by model, to quantify the compute saving vs pure-heavy decode."""
    heavy_forwards: int = 0
    draft_forwards: int = 0
    draft_commits: int = 0      # tokens committed by the drafter (the speculative wins)
    heavy_commits: int = 0      # tokens committed by the heavy


@torch.no_grad()
def decode_block_dbet(model, x, bs, be, attn, heavy_threshold, draft_threshold,
                      max_iters, max_draft_iters, tau, stats):
    """Decode one block in place via alternating heavy-commit / drafter-extend. Returns nothing (mutates x,
    updates stats). bs/be = block start/end (grid-aligned); attn = block-causal 4D mask over [0,be)."""
    active = (x[0:1, bs:be] == MASK_ID)                                 # original decode region
    block_logits = None

    it = 0
    while it < max_iters and bool((x[0:1, bs:be] == MASK_ID).any()):
        # ---- heavy forward + decode_uniform commit ----
        signals = model.extract_heavy_signals(x[:, :be], attn)
        stats.heavy_forwards += 1
        block_logits = signals["logits"][:, bs:be]                     # [1, blk, V]
        mask_idx = (x[0:1, bs:be] == MASK_ID)
        x0, high_conf_idx, _, brk = dmax_commit_uniform(block_logits, mask_idx, active, heavy_threshold)
        n_before = int(mask_idx.sum())
        if high_conf_idx.any():
            nb = x[0, bs:be].clone()
            nb[high_conf_idx[0]] = x0[0][high_conf_idx[0]]
            x[0, bs:be] = nb
            stats.heavy_commits += int(high_conf_idx.sum())
        if not bool((x[0:1, bs:be] == MASK_ID).any()):
            break

        # ---- drafter extend (1..max_draft_iters passes, confidence-gated commits) ----
        for _di in range(max_draft_iters):
            if not bool((x[0:1, bs:be] == MASK_ID).any()):
                break
            # relabel prefix/canvas from the UPDATED x (heavy hidden reused as prefix conditioning)
            signals["input_ids"] = x[:, :be]
            signals["prefix_idx"], signals["canvas_idx"] = model._split_prefix_denoise(x[:, :be])
            d = model.draft_forward(signals, attention_mask=None, tau=tau)
            stats.draft_forwards += 1
            dlogits, dconf = d["logits"], d["conf"]                     # [1,C,V], [1,C] (C = #masked in block)
            if dconf is None:                                          # no conf head -> can't gate; bail to heavy
                break
            tokens, commit = draft_commit_confident(dlogits, dconf, draft_threshold)
            # map canvas (contiguous right part of block) back to block positions
            canvas_pos = (x[0, bs:be] == MASK_ID).nonzero(as_tuple=True)[0]   # block-local indices, ascending
            sel = canvas_pos[commit]
            if sel.numel() == 0:
                break
            nb = x[0, bs:be].clone()
            nb[sel] = tokens[commit]
            x[0, bs:be] = nb
            stats.draft_commits += int(sel.numel())

        n_after = int((x[0:1, bs:be] == MASK_ID).sum())
        if n_after == n_before:                                        # neither model progressed -> force one token
            mask_pos = (x[0, bs:be] == MASK_ID).nonzero(as_tuple=True)[0]
            if mask_pos.numel() > 0:
                p = int(mask_pos[0])
                x[0, bs + p] = int(block_logits[0, p].argmax())
        it += 1

    # safety: never leave a [MASK] in the output
    still = (x[0:1, bs:be] == MASK_ID)
    if still.any() and block_logits is not None:
        fill = block_logits[0].argmax(dim=-1)
        nb = x[0, bs:be].clone()
        nb[still[0]] = fill[still[0]]
        x[0, bs:be] = nb


@torch.no_grad()
def generate_dbet(model, prompt_ids, gen_length, block_length,
                  heavy_threshold=0.9, draft_threshold=0.7, max_iter_per_block=32,
                  max_draft_iters=1, tau=None, early_stop=True):
    """Grid-aligned multi-block DBet generation. Returns (response_ids [n], DbetGenerateStats); response_ids
    excludes the prompt and is cut at the first EOS.
    heavy_threshold: decode_uniform commit confidence for the HEAVY (DMax default 0.9 here for high precision).
    draft_threshold: the trained confidence-head gate for committing DRAFTER tokens (higher = safer/slower)."""
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
    for b in range(num_blocks):
        bs = first_block_start + b * block_length
        be = bs + block_length
        attn = build_block_causal_mask(be, block_length, dtype=torch.bfloat16, device=device)
        decode_block_dbet(model, x, bs, be, attn, heavy_threshold, draft_threshold,
                          max_iter_per_block, max_draft_iters, tau, stats)
        if early_stop:
            resp_lo = max(P, bs)
            seg = x[0, resp_lo:be]
            eos_pos = (seg == EOS_ID).nonzero(as_tuple=True)[0]
            if eos_pos.numel() > 0:
                eos_cut = resp_lo + int(eos_pos[0].item())
                if be < L:
                    x[0, be:] = PAD_ID
                break

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
        max_draft_iters=args.max_draft_iters)
    text = tokenizer.decode(response_ids, skip_special_tokens=True)
    print(f"[dbet] heavy_fwd={stats.heavy_forwards} draft_fwd={stats.draft_forwards} "
          f"heavy_commits={stats.heavy_commits} draft_commits={stats.draft_commits}")
    print(f"[dbet] answer: {text!r}")


if __name__ == "__main__":
    _smoke_test()
