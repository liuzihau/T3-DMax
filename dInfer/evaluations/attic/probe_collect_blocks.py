# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# PREMISE PROBE — Variant B, stage 1: block-structured collection.
#
# The per-position probe (probe_collect_t3d.py + probe_fit.py) showed the static
# anchor does NOT per-position encode the converged token. But T3-D's talk is a
# BLOCK-LEVEL model that sees the revealed neighbors -- the per-position probe is
# blind to exactly that context. Variant B tests the fair question: can a
# LIGHTWEIGHT block-level model, given the static anchor + revealed neighbors,
# recover the converged tokens (especially on the flip positions)?
#
# This collector re-runs the frozen full model's canonical decode and saves, per
# decode BLOCK (not per flattened position):
#   anchor : iter-0 last hidden for ALL block positions   [n_blocks, block_len, D]
#   y      : converged block (prompt-tail + committed)     [n_blocks, block_len]
#   arg0   : iter-0 lm_head argmax for all block positions [n_blocks, block_len]
#   decode_mask : which positions were originally MASK     [n_blocks, block_len] bool
#   group  : prompt index (prompt-disjoint split downstream)[n_blocks]
#
# Reuses load_full_model/_think + the decode loop from probe_collect_t3d.

import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_DINFER_PYTHON = os.path.abspath(os.path.join(_HERE, "..", "python"))
if _DINFER_PYTHON not in sys.path:
    sys.path.insert(0, _DINFER_PYTHON)

from transformers import AutoTokenizer  # noqa: E402

from dinfer.decoding.generate_t3d import (  # noqa: E402
    EOS_ID, MASK_ID, build_block_causal_mask, dmax_commit_uniform,
)
from probe_collect_t3d import GSM8K_USER_TEMPLATE, _think, load_full_model  # noqa: E402


@torch.no_grad()
def collect_blocks_for_prompt(model, prompt_ids, gen_length, block_length, threshold,
                              max_iters_per_block, max_blocks):
    """Returns lists of per-block tensors: (anchor[blk,D], y[blk], arg0[blk], decode_mask[blk])."""
    device = prompt_ids.device
    P = prompt_ids.shape[1]
    first_block_start = (P // block_length) * block_length
    end_target = P + gen_length
    num_blocks = min((end_target - first_block_start + block_length - 1) // block_length, max_blocks)
    L = first_block_start + num_blocks * block_length

    x = torch.full((1, L), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids

    anchors, ys, arg0s, dmasks = [], [], [], []
    for b in range(num_blocks):
        bs = first_block_start + b * block_length
        be = bs + block_length
        attn = build_block_causal_mask(be, block_length, dtype=torch.bfloat16, device=device)
        active_index = (x[0:1, bs:be] == MASK_ID)
        if not active_index.any():
            continue

        anchor_block = arg0_block = last_logits = None
        for it in range(max_iters_per_block + 1):
            hidden = _think(model, x[:, :be], attn)
            block_hidden = hidden[:, bs:be]                       # [1, blk, D]
            logits_block = model.lm_head(block_hidden)
            last_logits = logits_block
            if it == 0:
                anchor_block = block_hidden[0].float().cpu()       # [blk, D]
                arg0_block = logits_block[0].argmax(dim=-1).cpu()  # [blk]

            mask_idx = (x[0:1, bs:be] == MASK_ID)
            x0, high_conf_idx, max_probs, breakflag = dmax_commit_uniform(
                logits_block, mask_idx, active_index, threshold)
            update_mask = high_conf_idx                            # freeze-after-commit
            changed = update_mask & (x0 != x[0:1, bs:be])
            if update_mask.any():
                nb = x[0, bs:be].clone()
                nb[update_mask[0]] = x0[0][update_mask[0]]
                x[0, bs:be] = nb
            if bool(breakflag) or (not changed.any()) or (not (x[0:1, bs:be] == MASK_ID).any()):
                break

        leftover = (x[0:1, bs:be] == MASK_ID)
        if leftover.any() and last_logits is not None:
            fill = last_logits[0].argmax(dim=-1)
            nb = x[0, bs:be].clone()
            nb[leftover[0]] = fill[leftover[0]]
            x[0, bs:be] = nb

        anchors.append(anchor_block)                              # [blk, D]
        ys.append(x[0, bs:be].cpu())                              # [blk]
        arg0s.append(arg0_block)                                  # [blk]
        dmasks.append(active_index[0].cpu())                      # [blk] bool

        resp_lo = max(P, bs)
        if (x[0, resp_lo:be] == EOS_ID).any():
            break

    return anchors, ys, arg0s, dmasks


def main():
    p = argparse.ArgumentParser(description="Premise probe Variant B — block-structured collection")
    p.add_argument("--model_path", required=True)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--out_path", required=True)
    p.add_argument("--num_prompts", type=int, default=200)
    p.add_argument("--gen_length", type=int, default=256)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--max_iters_per_block", type=int, default=32)
    p.add_argument("--max_blocks", type=int, default=8)
    p.add_argument("--gt_jsonl_path", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    tok_path = args.tokenizer_path or args.model_path
    tok_path = os.path.abspath(tok_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    model = load_full_model(args.model_path, args.device)

    if args.gt_jsonl_path:
        rows = []
        with open(args.gt_jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        rows = [{"question": r["question"]} for r in ds]
    rows = rows[: args.num_prompts]

    print(f"[probe-blocks] model={args.model_path}  {len(rows)} prompts  block={args.block_length}")

    A, Y, ARG0, DM, GROUP = [], [], [], [], []
    t0 = time.time()
    for i, row in enumerate(rows):
        messages = [{"role": "user", "content": GSM8K_USER_TEMPLATE.format(question=row["question"])}]
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(args.device)
        anchors, ys, arg0s, dmasks = collect_blocks_for_prompt(
            model, prompt_ids, args.gen_length, args.block_length, args.threshold,
            args.max_iters_per_block, args.max_blocks)
        for anchor_block, y_block, arg0_block, dmask in zip(anchors, ys, arg0s, dmasks):
            A.append(anchor_block); Y.append(y_block); ARG0.append(arg0_block)
            DM.append(dmask); GROUP.append(i)
        if i < 3 or (i + 1) % 25 == 0:
            print(f"[{i+1}/{len(rows)}] blocks so far={len(A)}")

    anchor = torch.stack(A, 0).half()                 # [n_blocks, blk, D]
    y = torch.stack(Y, 0)                              # [n_blocks, blk]
    arg0 = torch.stack(ARG0, 0)                        # [n_blocks, blk]
    decode_mask = torch.stack(DM, 0)                   # [n_blocks, blk] bool
    group = torch.tensor(GROUP, dtype=torch.long)      # [n_blocks]

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    torch.save({"anchor": anchor, "y": y, "arg0": arg0, "decode_mask": decode_mask,
                "group": group, "mask_id": MASK_ID,
                "meta": {"model_path": args.model_path, "block_length": args.block_length,
                         "d_model": anchor.shape[-1], "n_blocks": anchor.shape[0]}}, args.out_path)
    n_dec = int(decode_mask.sum())
    flips = int(((arg0 != y) & decode_mask).sum())
    print(f"[probe-blocks] saved {anchor.shape[0]} blocks (D={anchor.shape[-1]}) -> {args.out_path}")
    print(f"[probe-blocks] decode positions={n_dec}  flip_rate={flips/max(n_dec,1):.1%}")
    print(f"[probe-blocks] elapsed {time.time()-t0:.1f}s. Next: probe_fit_b.py --data {args.out_path}")


if __name__ == "__main__":
    main()
