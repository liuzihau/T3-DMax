# Copyright 2026 University of Sydney. Apache-2.0.
"""Isolate WHY the DBet cache isn't bit-exact. Three controlled comparisons on the prompt-prefix [0,upto):
  A DETERMINISM : same forward twice (use_cache=False)      -> nonzero => fused-MoE / kernel non-determinism
  B LENGTH      : forward [0,upto) vs [0,be)[:upto]         -> nonzero => hidden depends on total length
                                                               (MoE capacity/batch routing, NOT block-causal)
  C CACHE FLAG  : [0,upto) use_cache=True vs False          -> nonzero => the cache code path changes compute
All three use HARD embeds + the same block-causal mask, so only the controlled variable changes.

  python evaluations/diag_determinism.py --drafter_path <hf_ckpt> --heavy_path <DMax> --block_length 32
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "python")))
from dinfer.decoding.generate_dbet import build_block_causal_mask, load_dbet_model, _new_dynamic_cache, MASK_ID  # noqa: E402
from eval_dbet_gsm8k import load_gsm8k_test                                                                       # noqa: E402
from transformers import AutoTokenizer                                                                            # noqa: E402


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
    upto = (P // block) * block                          # prompt-prefix length (block-aligned)
    be = upto + block
    L = be + block
    x = torch.full((1, L), MASK_ID, dtype=torch.long, device=args.device); x[:, :P] = pid
    embed = model.draft.frozen_embed
    m_short = build_block_causal_mask(upto, block, dtype=torch.bfloat16, device=args.device)
    m_long = build_block_causal_mask(be, block, dtype=torch.bfloat16, device=args.device)
    print(f"P={P} upto={upto} be={be}")

    def fwd(n, mask, pkv=None, uc=False):
        return model.extract_heavy_signals(x[:, :n], attention_mask=mask, inputs_embeds=embed(x[:, :n]),
                                           past_key_values=pkv, use_cache=uc)

    def rep(name, a, b):
        d = (a.float() - b.float()).abs()
        print(f"  {name:34s} max|Δ|={d.max().item():.5f}  mean|Δ|={d.mean().item():.6f}")

    A1 = fwd(upto, m_short); A2 = fwd(upto, m_short)
    B_long = fwd(be, m_long)
    C = fwd(upto, m_short, pkv=_new_dynamic_cache(), uc=True)

    print("prompt-prefix [0,upto) h_sel comparisons:")
    rep("A determinism (same fwd x2)", A1["h_sel"], A2["h_sel"])
    rep("B length ([0,upto] vs [0,be][:upto])", A1["h_sel"], B_long["h_sel"][:, :upto])
    rep("C cache=True vs False", C["h_sel"], A1["h_sel"])
    print("\nA>0 -> fused-MoE non-determinism (bit-exact impossible; validate on accuracy).")
    print("B>0 -> hidden depends on total length (MoE capacity/batch routing) -> prompt-cache can't match no-cache.")
    print("C>0 -> the use_cache code path itself changes compute.")


if __name__ == "__main__":
    main()
