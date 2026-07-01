# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# DBet GSM8K decode driver (standalone; DMax-style, not lm-eval-harness).
# Mirrors eval_t3d_gsm8k.py: generate -> predictions jsonl -> grade with val_gsm8k.py.
# Uses the canonical decode in dinfer.decoding.generate_dbet (heavy decode_uniform
# commit + confidence-gated drafter extend). Plain PyTorch, single GPU.
#
# Output jsonl, one row per GSM8K test example IN ORDER:
#   {"answer", "question", "heavy_forwards", "draft_forwards", "heavy_commits", "draft_commits", "gen_tokens"}
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

from dinfer.decoding.generate_dbet import generate_dbet, load_dbet_model  # noqa: E402

# verbatim from DMax's gsm8k-llada-mini.yaml doc_to_text (matches eval_t3d_gsm8k.py)
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
    p = argparse.ArgumentParser(description="DBet GSM8K decode (heavy + confidence-gated drafter)")
    p.add_argument("--drafter_path", required=True, help="DBet drafter hf_ckpt (drafter-only weights).")
    p.add_argument("--heavy_path", required=True, help="DMax-Math-16B-moe-merge checkpoint (frozen heavy).")
    p.add_argument("--tokenizer_path", default=None, help="default: heavy_path.")
    p.add_argument("--out_path", required=True)
    p.add_argument("--gen_length", type=int, default=256)
    p.add_argument("--block_length", type=int, default=32, help="MUST match training (32).")
    p.add_argument("--heavy_threshold", type=float, default=0.9, help="heavy decode_uniform commit confidence.")
    p.add_argument("--draft_threshold", type=float, default=0.7, help="trained conf-head gate for drafter commits.")
    p.add_argument("--max_iters_per_block", type=int, default=32)
    p.add_argument("--max_draft_iters", type=int, default=1, help="drafter passes per heavy forward.")
    p.add_argument("--heavy_only", action="store_true", help="pure-DMax baseline (no drafter) via the same harness.")
    p.add_argument("--heavy_tau", type=float, default=1.0, help="soft-embed temperature for heavy commits.")
    p.add_argument("--heavy_top_k", type=int, default=1, help="soft-embed top-k for heavy commits.")
    p.add_argument("--draft_tau", type=float, default=1.0, help="soft-embed temperature for drafter commits.")
    p.add_argument("--draft_top_k", type=int, default=1, help="soft-embed top-k for drafter commits.")
    p.add_argument("--no_early_stop", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gt_jsonl_path", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    tok_path = os.path.abspath(args.tokenizer_path or args.heavy_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    model = load_dbet_model(args.drafter_path, args.heavy_path, args.device)
    print(f"[gsm8k-dbet] block={args.block_length} gen={args.gen_length} "
          f"heavy_thr={args.heavy_threshold} draft_thr={args.draft_threshold} max_draft_iters={args.max_draft_iters}")

    rows = load_gsm8k_test(limit=args.limit, gt_jsonl_path=args.gt_jsonl_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    print(f"[gsm8k-dbet] decoding {len(rows)} examples -> {args.out_path}")

    t0 = time.time()
    tot_h, tot_d, tot_hc, tot_dc = 0, 0, 0, 0
    tot_wall, tot_htime, tot_dtime, tot_tok = 0.0, 0.0, 0.0, 0
    with open(args.out_path, "w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            messages = [{"role": "user", "content": GSM8K_USER_TEMPLATE.format(question=row["question"])}]
            prompt_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(args.device)
            response_ids, stats = generate_dbet(
                model, prompt_ids,
                gen_length=args.gen_length, block_length=args.block_length,
                heavy_threshold=args.heavy_threshold, draft_threshold=args.draft_threshold,
                max_iter_per_block=args.max_iters_per_block, max_draft_iters=args.max_draft_iters,
                early_stop=not args.no_early_stop, use_draft=not args.heavy_only,
                heavy_tau=args.heavy_tau, heavy_top_k=args.heavy_top_k,
                draft_tau=args.draft_tau, draft_top_k=args.draft_top_k)
            text = tokenizer.decode(response_ids, skip_special_tokens=True)
            tot_h += stats.heavy_forwards; tot_d += stats.draft_forwards
            tot_hc += stats.heavy_commits; tot_dc += stats.draft_commits
            tot_wall += stats.wall_time; tot_htime += stats.heavy_time; tot_dtime += stats.draft_time
            tot_tok += int(response_ids.shape[0])
            fh.write(json.dumps({
                "answer": text, "question": row["question"],
                "heavy_forwards": stats.heavy_forwards, "draft_forwards": stats.draft_forwards,
                "heavy_commits": stats.heavy_commits, "draft_commits": stats.draft_commits,
                "wall_time": round(stats.wall_time, 4), "heavy_time": round(stats.heavy_time, 4),
                "draft_time": round(stats.draft_time, 4), "gen_tokens": int(response_ids.shape[0]),
            }, ensure_ascii=False) + "\n")
            fh.flush()
            if i < 3 or (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(rows)}] heavy={stats.heavy_forwards} draft={stats.draft_forwards} "
                      f"draft_commits={stats.draft_commits} tail={text[-160:]!r}")

    dt = time.time() - t0
    n = max(len(rows), 1)
    frac = tot_dc / max(tot_dc + tot_hc, 1)
    mode = "HEAVY-ONLY" if args.heavy_only else "DBet"
    print(f"[gsm8k-dbet] {mode} done in {dt:.1f}s. mean heavy/ex={tot_h/n:.1f} draft/ex={tot_d/n:.1f} "
          f"| tokens committed by drafter: {frac:.1%}")
    print(f"[gsm8k-dbet] mean decode wall/ex={tot_wall/n:.2f}s (heavy {tot_htime/n:.2f}s + draft {tot_dtime/n:.2f}s) "
          f"| throughput={tot_tok / max(tot_wall, 1e-6):.1f} tok/s "
          f"| avg {1e3*tot_htime/max(tot_h,1):.0f}ms/heavy-fwd {1e3*tot_dtime/max(tot_d,1):.0f}ms/draft-fwd")
    grader = os.path.join(_HERE, "val_gsm8k.py")
    print(f"[gsm8k-dbet] grade: python {grader} --pred-path {args.out_path}"
          + (f" --limit {args.limit}" if args.limit else ""))


if __name__ == "__main__":
    main()
