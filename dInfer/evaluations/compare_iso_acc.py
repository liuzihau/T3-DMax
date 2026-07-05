# Copyright 2026 University of Sydney. Apache-2.0.
"""Per-example iso-accuracy diff between two graded runs (val_gsm8k *_eval_details_solution.jsonl files).
Same accuracy COUNT (e.g. 90=90) is weak evidence; this checks whether they get the SAME questions right.

  python evaluations/compare_iso_acc.py \
    --a report_g512_full/preds_dbet_h0.9_d0.9_k2_eval_details_solution.jsonl --name-a dbet \
    --b report_g512_full/preds_heavy_h0.9_eval_details_solution.jsonl        --name-b heavy
"""
import argparse
import json


def load(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["index"]] = bool(r["correct"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="details jsonl for run A (e.g. DBet)")
    ap.add_argument("--b", required=True, help="details jsonl for run B (e.g. heavy)")
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    ap.add_argument("--show", type=int, default=20, help="how many disagreement indices to print")
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    idx = sorted(set(a) & set(b))
    n = len(idx)
    if n == 0:
        print("no overlapping indices"); return
    both = sum(1 for i in idx if a[i] and b[i])
    neither = sum(1 for i in idx if not a[i] and not b[i])
    a_only = [i for i in idx if a[i] and not b[i]]
    b_only = [i for i in idx if not a[i] and b[i]]
    agree = both + neither

    print(f"n={n}   acc_{args.name_a}={sum(a[i] for i in idx)/n:.1%}   acc_{args.name_b}={sum(b[i] for i in idx)/n:.1%}")
    print(f"both correct        : {both:5d} ({both/n:6.1%})")
    print(f"both wrong          : {neither:5d} ({neither/n:6.1%})")
    print(f"{args.name_a}-only correct  : {len(a_only):5d} ({len(a_only)/n:6.1%})   ({args.name_b} missed these)")
    print(f"{args.name_b}-only correct  : {len(b_only):5d} ({len(b_only)/n:6.1%})   ({args.name_a} missed these)")
    print(f"AGREEMENT (same right/wrong): {agree}/{n} = {agree/n:.1%}")
    print(f"[iso-accuracy verdict] {'IDENTICAL behavior' if not a_only and not b_only else 'same count but DIFFERENT examples -> not per-example identical'}")
    if a_only:
        print(f"  {args.name_a}-only indices: {a_only[:args.show]}{' ...' if len(a_only) > args.show else ''}")
    if b_only:
        print(f"  {args.name_b}-only indices: {b_only[:args.show]}{' ...' if len(b_only) > args.show else ''}")


if __name__ == "__main__":
    main()
