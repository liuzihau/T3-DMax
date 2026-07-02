#!/usr/bin/env python3
# Copyright 2026 University of Sydney. Apache-2.0.
#
# Diagnose GSM8K predictions (from eval_dbet_gsm8k.py): gen-token distribution, cap-hit rate,
# and DEGENERATE-tail rate (the "\n\n\n..."/"000..."/trailing-spaces collapse). Tells you whether
# a config's extra tokens are real content or junk. Reads only the preds jsonl (stdlib only).
#
#   python evaluations/analyze_gen_lengths.py --dir sweep_out_g1024 [--near_cap 1000]

import argparse
import glob
import json
import os
import re
import statistics as st

_RUN = re.compile(r"(.)\1{19,}", re.DOTALL)   # 20+ repeats of any single char (incl. newline)


def degenerate_tail(text, tail_len=80):
    """True if the answer ends in a degenerate run (all whitespace, a long single-char run, or <=2 unique
    non-space chars) -- the diffusion-decode collapse we saw (trailing '\\n'/'0'/spaces, no clean answer)."""
    t = text or ""
    if not t.strip():
        return True
    tail = t[-tail_len:]                            # raw tail (do NOT pre-strip: a newline/space run IS the signal)
    if tail.strip() == "":
        return True
    if _RUN.search(tail):
        return True
    uniq = set(tail.replace(" ", "").replace("\n", "").replace("\t", ""))
    return len(tail) >= 40 and len(uniq) <= 2


def summarize(path, near_cap):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        return None
    toks = [int(r.get("gen_tokens", 0)) for r in rows]
    capped = sum(1 for t in toks if t >= near_cap)
    degen = sum(1 for r in rows if degenerate_tail(r.get("answer", "")))
    hf = [r["heavy_forwards"] for r in rows if "heavy_forwards" in r]
    df = [r["draft_forwards"] for r in rows if "draft_forwards" in r]
    wt = [r["wall_time"] for r in rows if "wall_time" in r]
    return {
        "name": os.path.basename(path).replace("preds_", "").replace(".jsonl", ""),
        "n": len(rows), "mean_tok": st.mean(toks), "med_tok": st.median(toks), "max_tok": max(toks),
        "cap_pct": 100.0 * capped / len(rows), "degen_pct": 100.0 * degen / len(rows),
        "heavy": st.mean(hf) if hf else float("nan"), "draft": st.mean(df) if df else float("nan"),
        "wall": st.mean(wt) if wt else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description="gen-token + degenerate-tail diagnostic for GSM8K preds")
    ap.add_argument("--dir", default="./sweep_out_g1024")
    ap.add_argument("--glob", default="preds_*.jsonl")
    ap.add_argument("--near_cap", type=int, default=1000, help="gen_tokens >= this = likely truncated / no-EOS")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, args.glob)))
    if not files:
        print(f"no files match {os.path.join(args.dir, args.glob)}")
        return
    hdr = (f"{'config':<26}{'n':>5}{'mean_tok':>9}{'med':>6}{'max':>6}{'cap%':>7}{'degen%':>8}"
           f"{'heavy/ex':>10}{'draft/ex':>10}{'wall/ex':>9}")
    print(hdr)
    print("-" * len(hdr))
    for f in files:
        s = summarize(f, args.near_cap)
        if s:
            print(f"{s['name']:<26}{s['n']:>5}{s['mean_tok']:>9.0f}{s['med_tok']:>6.0f}{s['max_tok']:>6}"
                  f"{s['cap_pct']:>6.0f}%{s['degen_pct']:>7.0f}%{s['heavy']:>10.1f}{s['draft']:>10.1f}"
                  f"{s['wall']:>9.2f}")


if __name__ == "__main__":
    main()
