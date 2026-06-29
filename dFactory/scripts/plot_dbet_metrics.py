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

    # ---- 1. loss + accuracy, two panels: train (left) | val (right) ----
    def _panel(ax, recs, smoothed, title):
        """loss on the left y-axis; acc + acc6 (first-T) on the right y-axis."""
        xs, ls = _series(recs, "loss")
        style = dict(lw=1.8) if smoothed else dict(lw=1.5, marker="o", ms=4)
        if xs:
            if smoothed:
                ax.plot(xs, ls, color="tab:blue", alpha=0.2, lw=0.8)
                ls = _ema(ls, args.ema)
            ax.plot(xs, ls, color="tab:blue", label="loss", **style)
        ax.set_xlabel("step"); ax.set_ylabel("loss", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        axr = ax.twinx()
        for key, color, lab in [("acc", "tab:green", "acc (all remaining)"),
                                ("acc6", "tab:orange", "acc@T (first-T)")]:
            kx, ky = _series(recs, key)
            if kx:
                if smoothed:
                    ky = _ema(ky, args.ema)
                axr.plot(kx, ky, color=color, label=lab, **style)
        axr.set_ylabel("accuracy", color="tab:green")
        axr.tick_params(axis="y", labelcolor="tab:green"); axr.set_ylim(0, 1)
        lines = [l for l in ax.get_lines() + axr.get_lines() if not l.get_label().startswith("_")]
        ax.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
        ax.set_title(title)
        return axr

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6))
    arL = _panel(axL, train, smoothed=True, title="Training")
    arR = _panel(axR, val, smoothed=False, title="Validation")
    # share the accuracy axis range across panels for easy comparison
    hi = max([arL.get_ylim()[1], arR.get_ylim()[1], 1.0])
    arL.set_ylim(0, hi); arR.set_ylim(0, hi)
    fig.suptitle("DBet: loss & drafter accuracy", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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

        # ---- top-1/2/3 coverage + heavy 2nd-pass vs block position ----
        ptk = last.get("acc_by_pos_topk", [])        # [[k, top1, top2, top3, h2, count], ...]
        if ptk:
            ks = [r[0] for r in ptk]
            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.plot(ks, [r[1] for r in ptk], "o-", color="tab:blue", lw=1.6, ms=3, label="drafter top-1")
            ax.plot(ks, [r[2] for r in ptk], "o-", color="tab:green", lw=1.4, ms=3, label="drafter top-2")
            ax.plot(ks, [r[3] for r in ptk], "o-", color="tab:olive", lw=1.4, ms=3, label="drafter top-3")
            if any(r[4] > 0 for r in ptk):
                ax.plot(ks, [r[4] for r in ptk], "s--", color="tab:red", lw=1.4, ms=3, label="heavy 2nd pass")
            ax.set_xlabel("distance into block (remaining-position index k)")
            ax.set_ylabel("accuracy / coverage"); ax.set_ylim(0, 1)
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            ax.set_title(f"DBet: top-k coverage & heavy 2nd-pass by position (step {last['step']})")
            _save(fig, out_dir, "topk_by_position")

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

        # ---- accuracy heatmap over (mask ratio sigma, block position k) ----
        grid = last.get("acc_by_pos_sigma", [])
        if grid:
            import numpy as np
            sigmas = sorted({row[0] for row in grid})
            ks = sorted({row[1] for row in grid})
            si = {s: i for i, s in enumerate(sigmas)}
            ki = {k: i for i, k in enumerate(ks)}
            acc = np.full((len(ks), len(sigmas)), np.nan)        # rows = k, cols = sigma; empty cells stay NaN
            cnt = np.zeros((len(ks), len(sigmas)))
            for s, k, a, n in grid:
                acc[ki[k], si[s]] = a
                cnt[ki[k], si[s]] = n
            masked = np.ma.masked_invalid(acc)                   # empty (no-data) cells -> "bad" color
            fig, ax = plt.subplots(figsize=(1.4 * len(sigmas) + 2.5, 0.32 * len(ks) + 2))
            cmap = plt.cm.viridis.copy(); cmap.set_bad("lightgray")
            im = ax.imshow(masked, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks(range(len(sigmas))); ax.set_xticklabels([f"{s:g}" for s in sigmas])
            ax.set_yticks(range(len(ks))); ax.set_yticklabels([str(k) for k in ks])
            ax.set_xlabel("mask ratio sigma"); ax.set_ylabel("block position index k")
            ax.set_title(f"DBet: drafter accuracy by (sigma, position)  step {last['step']}\n(grey = no data)")
            for r in range(len(ks)):
                for c in range(len(sigmas)):
                    if not masked.mask[r, c]:
                        ax.text(c, r, f"{acc[r, c]:.2f}", ha="center", va="center", fontsize=6,
                                color="white" if acc[r, c] < 0.55 else "black")
            fig.colorbar(im, ax=ax, label="accuracy", fraction=0.046, pad=0.04)
            _save(fig, out_dir, "acc_heatmap")

    print("done.")


if __name__ == "__main__":
    main()
