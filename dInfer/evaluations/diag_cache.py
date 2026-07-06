# Copyright 2026 University of Sydney. Apache-2.0.
"""Decisive localizer for the DBet prefix-KV cache. For ONE prompt it compares, at block 0's FIRST heavy
forward (no decode loop, no finalization), the cache-partial path vs the no-cache full path:
  - block logits            (partial forward correctness: mask / RoPE / KV)
  - block h_sel             (same, for the sel-layers)
  - prefix h_sel            (the prompt-cache correctness, vs no-cache's [0,bs) h_sel)
If all ~0 -> block-0 mechanics are bit-exact and the bug is the finalization/prefix hand-off (blocks 1+).
If block logits differ -> the partial forward itself (mask pass-through / cache) is wrong.

  python evaluations/diag_cache.py --drafter_path <hf_ckpt> --heavy_path <DMax> --block_length 32
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "python")))
from dinfer.decoding.generate_dbet import (build_block_causal_mask, load_dbet_model,     # noqa: E402
                                           _build_prefix_cache, MASK_ID)
from eval_dbet_gsm8k import load_gsm8k_test                                              # noqa: E402
from transformers import AutoTokenizer                                                   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drafter_path", required=True)
    p.add_argument("--heavy_path", required=True)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--gen_length", type=int, default=512)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(os.path.abspath(args.tokenizer_path or args.heavy_path), trust_remote_code=True)
    model = load_dbet_model(args.drafter_path, args.heavy_path, args.device)
    row = load_gsm8k_test(limit=1)[0]
    pid = tok.apply_chat_template([{"role": "user", "content": row["question"]}],
                                  add_generation_prompt=True, tokenize=True, return_tensors="pt").to(args.device)
    P = pid.shape[1]
    block = args.block_length
    first_block_start = (P // block) * block
    bs, be = first_block_start, first_block_start + block
    L = first_block_start + ((P + args.gen_length - first_block_start + block - 1) // block) * block
    x = torch.full((1, L), MASK_ID, dtype=torch.long, device=args.device); x[:, :P] = pid
    embed = model.draft.frozen_embed
    print(f"P={P} first_block_start={bs} be={be} block={block}")

    # ---- NO-CACHE full forward over [0,be) ----
    attn_full = build_block_causal_mask(be, block, dtype=torch.bfloat16, device=args.device)
    sig_f = model.extract_heavy_signals(x[:, :be], attention_mask=attn_full, inputs_embeds=embed(x[:, :be]))
    logits_f = sig_f["logits"][:, bs:be]          # [1,blk,V]
    hsel_blk_f = sig_f["h_sel"][:, bs:be]         # [1,blk,mD]
    hsel_pre_f = sig_f["h_sel"][:, :bs]           # [1,bs,mD]

    # ---- CACHE: build prompt-prefix [0,bs) then partial-forward the block ----
    cache, prefix_hsel = _build_prefix_cache(model, x, bs, block, 1.0, 1)
    blk = be - bs
    full_attend = torch.zeros(1, 1, blk, be, dtype=torch.bfloat16, device=args.device)
    sig_c = model.extract_heavy_signals(x[:, bs:be], attention_mask=full_attend, inputs_embeds=embed(x[:, bs:be]),
                                        past_key_values=cache, use_cache=True)
    logits_c = sig_c["logits"]                    # [1,blk,V]
    hsel_blk_c = sig_c["h_sel"]                   # [1,blk,mD]

    def rep(name, a, b):
        d = (a.float() - b.float()).abs()
        am = (a.argmax(-1) == b.argmax(-1)).float().mean().item() if a.dim() == 3 else float("nan")
        print(f"  {name:14s} max|Δ|={d.max().item():.5f}  mean|Δ|={d.mean().item():.6f}  argmax_match={am:.4f}")

    print("block-0 first forward, cache vs no-cache:")
    rep("block logits", logits_c, logits_f)
    rep("block h_sel", hsel_blk_c, hsel_blk_f)
    rep("prefix h_sel", prefix_hsel, hsel_pre_f)
    print("\nread: all max|Δ|~0 -> block-0 mechanics bit-exact (bug is finalization, blocks 1+).")
    print("      block logits differ -> partial-forward/mask/cache bug.  prefix h_sel differs -> prompt-cache bug.")


if __name__ == "__main__":
    main()
