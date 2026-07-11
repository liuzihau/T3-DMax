# Copyright 2026 University of Sydney. Apache-2.0.
# Aggregate the nightly baseline jsonls into one TSV: accuracy (via val_gsm8k.py subprocess) + per-example
# means. Handles both jsonl schemas: sglang/llada drivers write "forwards"; the eager DBet driver writes
# "heavy_forwards"/"draft_forwards".
#   python summarize_baseline.py --dir report_baseline_XXXX [--out summary.tsv]

import argparse
import glob
import json
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def grade(jsonl_path):
    """Run val_gsm8k.py and parse 'Accuracy: xx.xxxx%'."""
    try:
        out = subprocess.run([sys.executable, os.path.join(_HERE, "val_gsm8k.py"),
                              "--pred-path", jsonl_path],
                             capture_output=True, text=True, timeout=600).stdout
        m = re.search(r"Accuracy:\s*([\d.]+)%", out)
        return float(m.group(1)) if m else float("nan")
    except Exception as e:  # noqa: BLE001 - a broken run shouldn't kill the summary
        print(f"[warn] grading {jsonl_path} failed: {e}", file=sys.stderr)
        return float("nan")


def summarize(jsonl_path):
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return None
    n = len(rows)

    def mean(key):
        vals = [r[key] for r in rows if key in r]
        return sum(vals) / len(vals) if vals else float("nan")

    heavy = mean("heavy_forwards") if "heavy_forwards" in rows[0] else mean("forwards")
    draft = mean("draft_forwards") if "draft_forwards" in rows[0] else 0.0
    # draft_commits exists in BOTH schemas (eager + sglang) — the actual drafter activity signal
    dcommit = mean("draft_commits") if "draft_commits" in rows[0] else 0.0
    tok = mean("gen_tokens")
    wall = mean("wall_time")
    tps = tok / wall if wall and wall == wall and wall > 0 else float("nan")
    # tokens per drafter call -- the walk-quality decision number (v1 baseline ~1.3)
    tpc = dcommit / draft if draft and draft == draft and draft > 0 else float("nan")
    return {"n": n, "heavy/ex": heavy, "draft/ex": draft, "dcommit/ex": dcommit, "tok/call": tpc,
            "tok/ex": tok, "wall/ex": wall, "tok/s": tps}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--out", default=None, help="default: <dir>/summary.tsv")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.jsonl")))
    out_path = args.out or os.path.join(args.dir, "summary.tsv")
    cols = ["config", "n", "accuracy", "heavy/ex", "draft/ex", "dcommit/ex", "tok/call", "tok/ex", "wall/ex", "tok/s"]
    lines = ["\t".join(cols)]
    for f in files:
        s = summarize(f)
        if s is None:
            print(f"[warn] empty: {f}", file=sys.stderr)
            continue
        acc = grade(f)
        name = os.path.splitext(os.path.basename(f))[0]
        lines.append("\t".join([name, str(s["n"]), f"{acc:.2f}%", f"{s['heavy/ex']:.1f}",
                                f"{s['draft/ex']:.1f}", f"{s['dcommit/ex']:.1f}", f"{s['tok/call']:.2f}",
                                f"{s['tok/ex']:.0f}", f"{s['wall/ex']:.2f}", f"{s['tok/s']:.1f}"]))
        print(lines[-1])
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"[summary] -> {out_path}")


if __name__ == "__main__":
    main()
