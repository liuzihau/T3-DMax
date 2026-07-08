# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# T3-D GSM8K decode driver (standalone; DMax-style, not lm-eval-harness).
#
# Mirrors the role of DMax's eval flow: generate -> predictions jsonl -> grade
# with val_gsm8k.py. Uses the CANONICAL decode in dinfer.decoding.generate_t3d
# (soft-embedding decode_uniform, grid-aligned, multi-block). Plain PyTorch,
# single GPU. This is the ACCURACY path; throughput is not DMax-comparable until
# the vllm port.
#
# Output: predictions jsonl, one row per GSM8K test example IN ORDER:
#   {"answer": <text>, "question": ..., "think_forwards": ..., "talk_forwards": ...}
# Grade:  python val_gsm8k.py --pred-path <jsonl> [--limit N]

import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DINFER_PYTHON = os.path.abspath(os.path.join(_HERE, "..", "python"))   # .../dInfer/python
if _DINFER_PYTHON not in sys.path:
    sys.path.insert(0, _DINFER_PYTHON)

from transformers import AutoTokenizer  # noqa: E402

from dinfer.decoding.generate_t3d import generate_t3d, load_t3d_model  # noqa: E402

# GSM8K prompt — verbatim from DMax's gsm8k-llada-mini.yaml doc_to_text, wrapped
# in the chat template (DMax passes --apply_chat_template).
GSM8K_USER_TEMPLATE = "Question: {question}\nLet's think step by step\nAnswer:"


def load_gsm8k_test(limit=None, gt_jsonl_path=None):
    if gt_jsonl_path:
        rows = []
        with open(gt_jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        rows = [{"question": r["question"], "answer": r["answer"]} for r in ds]
    if limit is not None:
        rows = rows[:limit]
    return rows


def main():
    p = argparse.ArgumentParser(description="T3-D GSM8K decode (canonical paradigm)")
    p.add_argument("--model_path", required=True, help="T3-D hf_ckpt (ThinkTalk weights).")
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--out_path", required=True)
    p.add_argument("--gen_length", type=int, default=256)
    p.add_argument("--block_length", type=int, default=32, help="MUST match training (v6e/v2: 32).")
    p.add_argument("--threshold", type=float, default=0.3, help="DMax decode_uniform default.")
    p.add_argument("--max_iters_per_block", type=int, default=32)
    p.add_argument("--soft_top_k", type=int, default=1)
    p.add_argument("--no_early_stop", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gt_jsonl_path", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    tok_path = args.tokenizer_path or args.model_path
    tok_path = os.path.abspath(tok_path)   # HF rejects '..' paths; normalize unconditionally
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    model = load_t3d_model(args.model_path, args.device)
    print(f"[gsm8k-t3d] delta_head present: {model.delta_head is not None}  "
          f"block={args.block_length} gen={args.gen_length} threshold={args.threshold}")

    rows = load_gsm8k_test(limit=args.limit, gt_jsonl_path=args.gt_jsonl_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    print(f"[gsm8k-t3d] decoding {len(rows)} examples -> {args.out_path}")

    t0 = time.time()
    tot_think, tot_talk = 0, 0
    with open(args.out_path, "w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            messages = [{"role": "user", "content": GSM8K_USER_TEMPLATE.format(question=row["question"])}]
            prompt_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_tensors="pt",
            ).to(args.device)
            response_ids, stats = generate_t3d(
                model, prompt_ids,
                gen_length=args.gen_length, block_length=args.block_length,
                threshold=args.threshold, max_iter_per_block=args.max_iters_per_block,
                soft_top_k=args.soft_top_k, early_stop=not args.no_early_stop,
            )
            text = tokenizer.decode(response_ids, skip_special_tokens=True)
            tot_think += stats.think_forwards
            tot_talk += stats.talk_forwards
            fh.write(json.dumps({
                "answer": text, "question": row["question"],
                "think_forwards": stats.think_forwards,
                "talk_forwards": stats.talk_forwards,
                "gen_tokens": int(response_ids.shape[0]),
            }, ensure_ascii=False) + "\n")
            fh.flush()
            if i < 3 or (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(rows)}] think={stats.think_forwards} "
                      f"talk={stats.talk_forwards} answer_tail={text[-160:]!r}")

    dt = time.time() - t0
    n = max(len(rows), 1)
    print(f"[gsm8k-t3d] done in {dt:.1f}s. mean think/ex={tot_think/n:.1f} talk/ex={tot_talk/n:.1f}")
    grader = os.path.join(_HERE, "val_gsm8k.py")
    print(f"[gsm8k-t3d] grade: python {grader} --pred-path {args.out_path}"
          + (f" --limit {args.limit}" if args.limit else ""))


if __name__ == "__main__":
    main()
