# Copyright 2026 University of Sydney. Apache-2.0.
"""Acceptance gate for the DBet prefix-KV cache (--use_cache). Runs each GSM8K prompt BOTH ways
(no-cache vs cache) in one process and asserts the generated token ids are IDENTICAL. The cache is a
pure speed optimization, so any mismatch = a cache bug (crop timing / finalization / mask / position).

  python evaluations/validate_cache.py --drafter_path <hf_ckpt> --heavy_path <DMax> --limit 10 \
    --gen_length 512 --block_length 32 --heavy_threshold 0.9 --draft_threshold 0.9 --draft_top_k 2
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "python")))
from dinfer.decoding.generate_dbet import generate_dbet, load_dbet_model          # noqa: E402
from eval_dbet_gsm8k import load_gsm8k_test                                        # noqa: E402
from transformers import AutoTokenizer                                            # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drafter_path", required=True)
    p.add_argument("--heavy_path", required=True)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--gen_length", type=int, default=512)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--heavy_threshold", type=float, default=0.9)
    p.add_argument("--draft_threshold", type=float, default=0.9)
    p.add_argument("--heavy_top_k", type=int, default=1)
    p.add_argument("--draft_top_k", type=int, default=2)
    p.add_argument("--draft_committed_soft", action="store_true")
    p.add_argument("--no_draft_fix", action="store_true")
    p.add_argument("--gt_jsonl_path", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(os.path.abspath(args.tokenizer_path or args.heavy_path), trust_remote_code=True)
    model = load_dbet_model(args.drafter_path, args.heavy_path, args.device)
    rows = load_gsm8k_test(limit=args.limit, gt_jsonl_path=args.gt_jsonl_path)

    def gen(prompt_ids, use_cache):
        r, s = generate_dbet(
            model, prompt_ids, gen_length=args.gen_length, block_length=args.block_length,
            heavy_threshold=args.heavy_threshold, draft_threshold=args.draft_threshold,
            heavy_top_k=args.heavy_top_k, draft_top_k=args.draft_top_k,
            draft_committed_soft=args.draft_committed_soft, draft_fix=not args.no_draft_fix,
            use_cache=use_cache)
        return r, s

    n_ok = 0
    for i, row in enumerate(rows):
        msgs = [{"role": "user", "content": row["question"]}]
        pid = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt").to(args.device)
        r0, s0 = gen(pid, use_cache=False)
        r1, s1 = gen(pid, use_cache=True)
        same = (r0.shape == r1.shape) and bool(torch.equal(r0, r1))
        if same:
            n_ok += 1
            tag = "OK "
        else:
            # first divergence
            m = min(r0.shape[0], r1.shape[0])
            div = next((k for k in range(m) if int(r0[k]) != int(r1[k])), m)
            tag = f"MISMATCH @tok{div} (len {r0.shape[0]} vs {r1.shape[0]})"
        print(f"[{i+1}/{len(rows)}] {tag}  | heavy_fwd nocache={s0.heavy_forwards} cache={s1.heavy_forwards}")

    print(f"\n=== {n_ok}/{len(rows)} identical. "
          f"{'PASS — cache is correct, trust the timing.' if n_ok == len(rows) else 'FAIL — cache bug, do NOT trust timing.'} ===")


if __name__ == "__main__":
    main()
