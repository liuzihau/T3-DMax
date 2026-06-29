# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
"""Render DBet training/validation figures from the JSONL written by train_dbet.py (MetricsLogger).

    python scripts/plot_dbet_metrics.py --metrics ./dbet_outputs/dbet_metrics.jsonl --out ./dbet_outputs/figures

Produces (PNG + PDF):
  1. loss_accuracy   -- train loss (EMA) + train/val drafter accuracy over steps.
  2. confidence_auc  -- held-out confidence-head ROC-AUC over training (gate 0.7; prior probe 0.57).
  3. acc_by_position -- drafter accuracy vs distance-into-block (last eval).
  4. acc_by_sigma    -- accuracy & confidence-AUC vs mask ratio sigma (last eval).

Reads only the JSONL, so it works offline and is fully reproducible from the archived metrics file.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path):
    train, val = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            (train if r.get("split") == "train" else val).append(r)
    return train, val


def _ema(xs, beta=0.9):
    out, m = [], None
    for x in xs:
        m = x if m is None else beta * m + (1 - beta) * x
        out.append(m)
    return out


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def _series(records, key):
    xs = [r["step"] for r in records if key in r and r[key] == r[key]]
    ys = [r[key] for r in records if key in r and r[key] == r[key]]
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="dbet_metrics.jsonl path")
    ap.add_argument("--out", default=None, help="figures output dir (default: <metrics_dir>/figures)")
    ap.add_argument("--ema", type=float, default=0.9, help="EMA smoothing for noisy train curves")
    args = ap.parse_args()
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(args.metrics)), "figures")
    train, val = _load(args.metrics)
    print(f"loaded {len(train)} train + {len(val)} val records -> {out_dir}")

    # ---- 1. loss + accuracy ----
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ts, ls = _series(train, "loss")
    if ts:
        ax1.plot(ts, ls, color="tab:blue", alpha=0.25, lw=0.8)
        ax1.plot(ts, _ema(ls, args.ema), color="tab:blue", lw=1.8, label="train loss (EMA)")
    ax1.set_xlabel("step"); ax1.set_ylabel("loss", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ta, aa = _series(train, "acc")
    if ta:
        ax2.plot(ta, _ema(aa, args.ema), color="tab:green", lw=1.5, label="train acc (EMA)")
    va, vaa = _series(val, "acc")
    if va:
        ax2.plot(va, vaa, "o-", color="tab:red", lw=1.5, ms=4, label="val acc")
    ax2.set_ylabel("drafter accuracy (remaining)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green"); ax2.set_ylim(0, 1)
    lines = [l for l in ax1.get_lines() + ax2.get_lines() if not l.get_label().startswith("_")]
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
    ax1.set_title("DBet: loss & drafter accuracy")
    _save(fig, out_dir, "loss_accuracy")

    # ---- 2. confidence AUC over training ----
    xs, ys = _series(val, "auc")
    if xs:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(xs, ys, "o-", color="tab:purple", lw=1.8, ms=5, label="held-out conf-AUC")
        ax.axhline(0.7, ls="--", color="green", lw=1, label="ship gate (0.70)")
        ax.axhline(0.57, ls=":", color="gray", lw=1, label="prior probe (0.57)")
        ax.axhline(0.5, ls="-", color="lightgray", lw=0.8)
        ax.set_xlabel("step"); ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.45, 1.0)
        ax.legend(fontsize=8); ax.set_title("DBet: confidence-head AUC (heavy-acceptance)")
        _save(fig, out_dir, "confidence_auc")

    # ---- 3 & 4 from the LAST eval record that carries the sweep ----
    last = next((r for r in reversed(val) if "acc_by_pos" in r), None)
    if last:
        kp = last["acc_by_pos"]                      # [[k, acc, count], ...]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot([p[0] for p in kp], [p[1] for p in kp], "o-", color="tab:blue", lw=1.8, ms=4)
        ax.set_xlabel("distance into block (remaining-position index k)")
        ax.set_ylabel("drafter accuracy"); ax.set_ylim(0, 1)
        ax.set_title(f"DBet: accuracy vs block position (step {last['step']})")
        ax.grid(alpha=0.3)
        _save(fig, out_dir, "acc_by_position")

        sg = last.get("acc_by_sigma", [])
        ag = {s: a for s, a, _ in sg}
        au = {s: a for s, a in last.get("auc_by_sigma", [])}
        if sg:
            fig, ax = plt.subplots(figsize=(7, 4.2))
            ss = sorted(ag.keys())
            ax.plot(ss, [ag[s] for s in ss], "o-", color="tab:blue", lw=1.8, ms=5, label="accuracy")
            if au:
                ax.plot(sorted(au.keys()), [au[s] for s in sorted(au.keys())],
                        "s--", color="tab:purple", lw=1.5, ms=5, label="conf-AUC")
            ax.set_xlabel("mask ratio sigma"); ax.set_ylabel("value"); ax.set_ylim(0, 1)
            ax.legend(fontsize=8); ax.set_title(f"DBet: accuracy & AUC vs mask ratio (step {last['step']})")
            ax.grid(alpha=0.3)
            _save(fig, out_dir, "acc_by_sigma")

    print("done.")


if __name__ == "__main__":
    main()
