# Copyright 2026 University of Sydney. Apache-2.0.
"""Settle the B (length-dependence) cause: is it ATTENTION (prefix attends the current block) or MoE
capacity/batch routing (attention is block-causal but routing depends on the token set)?
Forward [0,be) TWICE with DIFFERENT content in the current block [upto,be) but IDENTICAL prefix [0,upto),
and compare the prefix hidden:
  Δ == 0  -> prefix is INDEPENDENT of the block content => attention is block-causal (prefix STABLE);
             B's length-dependence is MoE capacity/batch routing -> a prefix cache is *attention-valid*.
  Δ  > 0  -> prefix ATTENDS the block => FULL (non-block-causal) attention -> the "prefix" is not stable,
             a prefix KV cache is fundamentally INVALID for this decode.

  python evaluations/diag_attn.py --drafter_path <hf_ckpt> --heavy_path <DMax> --block_length 32
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "python")))
from dinfer.decoding.generate_dbet import build_block_causal_mask, load_dbet_model, MASK_ID   # noqa: E402
from eval_dbet_gsm8k import load_gsm8k_test                                                    # noqa: E402
from transformers import AutoTokenizer                                                         # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drafter_path", required=True)
    p.add_argument("--heavy_path", required=True)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(os.path.abspath(args.tokenizer_path or args.heavy_path), trust_remote_code=True)
    model = load_dbet_model(args.drafter_path, args.heavy_path, args.device)
    row = load_gsm8k_test(limit=1)[0]
    pid = tok.apply_chat_template([{"role": "user", "content": row["question"]}],
                                  add_generation_prompt=True, tokenize=True, return_tensors="pt").to(args.device)
    P = pid.shape[1]; block = args.block_length
    upto = (P // block) * block; be = upto + block
    embed = model.draft.frozen_embed
    m = build_block_causal_mask(be, block, dtype=torch.bfloat16, device=args.device)
    print(f"P={P} upto={upto} be={be}")

    x1 = torch.full((1, be), MASK_ID, dtype=torch.long, device=args.device); x1[:, :P] = pid  # block = MASK
    x2 = x1.clone(); x2[:, upto:be] = 100                                                     # block = other tokens
    assert torch.equal(x1[:, :upto], x2[:, :upto])                                            # prefix identical

    h1 = model.extract_heavy_signals(x1, attention_mask=m, inputs_embeds=embed(x1))["h_sel"][:, :upto]
    h2 = model.extract_heavy_signals(x2, attention_mask=m, inputs_embeds=embed(x2))["h_sel"][:, :upto]
    d = (h1.float() - h2.float()).abs()
    print(f"prefix [0,upto) h_sel, block=MASK vs block=other:  max|Δ|={d.max().item():.5f}  mean|Δ|={d.mean().item():.6f}")
    if d.max().item() < 1e-3:
        print("=> Δ~0: prefix INDEPENDENT of block -> attention is BLOCK-CAUSAL (prefix stable). B is MoE routing.")
        print("   The cache is attention-valid; fix MoE routing to be length-invariant (or validate on accuracy).")
    else:
        print("=> Δ>0: prefix ATTENDS the block -> FULL attention -> a prefix KV cache is fundamentally INVALID.")
        print("   Abandon the cache; the DBet-vs-heavy RATIO (same path) stays valid; align wall-clock via sglang.")


if __name__ == "__main__":
    main()
