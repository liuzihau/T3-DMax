# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# Plot the layer-readout probe (probe_layer_readout.py). Produces the go/no-go figures:
#   1. recall_vs_setsize.png   -- THE decision plot: per layer, masked nucleus-recall vs mean |S| (F1).
#   2. layer_recall_curves.png -- recall@K vs layer (masked), one panel per forward. "which layer locks in".
#   3. setsize_vs_layer.png    -- mean |S| + cap-saturation vs layer (masked, per forward).
#   4. heatmap_recall.png      -- layer x position recall@K (masked), one panel per forward.
#   5. position_bars_abs.png   -- x=position, revealed vs masked bars, one panel per K_ABS (chosen layer, F1).
#   6. position_bars_pct.png   -- same for K_PCT.
#   7. conditional_markov.png  -- conditional (prefix-all-correct) recall@K vs position vs unconditional.
#
# Run:  python probe_layer_plot.py --npz runs/probe_readout.npz [--layer L] [--recallK 10]

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MASKED, REVEALED = 0, 1


def _safe(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_dir", default=None, help="default: <npz dir>/plots")
    ap.add_argument("--layer", type=int, default=None, help="layer for the per-position bar plots; default=best")
    ap.add_argument("--recallK", type=int, default=10, help="K for the heatmap panel")
    args = ap.parse_args()

    d = np.load(args.npz)
    meta = json.load(open(os.path.splitext(args.npz)[0] + "_meta.json"))
    K_ABS, K_PCT, K_COND = meta["K_ABS"], meta["K_PCT"], meta["K_COND"]
    NL, F, Pn = d["count"].shape[:3]
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.npz)), "plots")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[plot] layers={NL} forwards={F} positions={Pn} agree={meta.get('agree'):.3f} -> {out_dir}")

    count = d["count"]                     # [NL,F,P,2]
    cnt_m = count[..., MASKED]             # [NL,F,P]
    # per-layer masked counts summed over positions (for layer-level curves)
    cnt_m_L = cnt_m.sum(axis=2)            # [NL,F]

    def masked_recall_abs(ki):             # [NL,F,P]
        return _safe(d["hit_abs"][ki, ..., MASKED], cnt_m)

    def masked_recall_abs_L(ki):           # [NL,F] recall over all masked positions
        return _safe(d["hit_abs"][ki, ..., MASKED].sum(axis=2), cnt_m_L)

    nuc_recall = _safe(d["hit_nuc"][..., MASKED], cnt_m)           # [NL,F]?  -> [NL,F,P]
    nuc_recall_L = _safe(d["hit_nuc"][..., MASKED].sum(axis=2), cnt_m_L)   # [NL,F]
    size_mean_L = _safe(d["size_sum"][..., MASKED].sum(axis=2), cnt_m_L)   # [NL,F]
    sat_rate_L = _safe(d["sat_sum"][..., MASKED].sum(axis=2), cnt_m_L)     # [NL,F]

    best_layer = args.layer if args.layer is not None else int(np.argmax(nuc_recall_L[:, 0]))
    layers = np.arange(NL)

    # ---- 1. recall vs set-size (THE decision plot), F1 ----
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(size_mean_L[:, 0], nuc_recall_L[:, 0], c=layers, cmap="viridis", s=60)
    for L in range(NL):
        ax.annotate(str(L), (size_mean_L[L, 0], nuc_recall_L[L, 0]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axhline(0.99, ls="--", c="r", lw=1, label="recall 0.99")
    ax.axvline(30, ls="--", c="gray", lw=1, label="|S|=30")
    ax.set_xlabel("mean |S| = |nucleus(0.5) ∩ top-200|  (masked, F1)")
    ax.set_ylabel("gold-in-S recall  (masked, F1)")
    ax.set_title("Prune validity: nucleus recall vs candidate-set size, per layer\n"
                 "GO if a layer sits top-left (high recall, small |S|)")
    plt.colorbar(sc, label="layer"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "recall_vs_setsize.png"), dpi=130); plt.close(fig)

    # ---- 2. recall@K vs layer, one panel per forward ----
    fig, axes = plt.subplots(1, F, figsize=(5 * F, 4), sharey=True, squeeze=False)
    for f in range(F):
        ax = axes[0][f]
        for ki, K in enumerate(K_ABS):
            ax.plot(layers, masked_recall_abs_L(ki)[:, f], marker="o", ms=3, label=f"@{K}")
        ax.plot(layers, nuc_recall_L[:, f], marker="s", ms=3, ls="--", c="k", label="nucleus∩200")
        ax.set_title(f"Forward {f+1}"); ax.set_xlabel("layer"); ax.grid(alpha=0.3)
        if f == 0:
            ax.set_ylabel("masked recall")
    axes[0][-1].legend(fontsize=7, ncol=2)
    fig.suptitle("Gold-token recall vs layer (masked positions)"); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "layer_recall_curves.png"), dpi=130); plt.close(fig)

    # ---- 3. set size + saturation vs layer ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for f in range(F):
        ax.plot(layers, size_mean_L[:, f], marker="o", ms=3, label=f"|S| F{f+1}")
    ax.set_xlabel("layer"); ax.set_ylabel("mean |S| (masked)"); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    for f in range(F):
        ax2.plot(layers, sat_rate_L[:, f], marker="x", ms=3, ls=":", label=f"sat F{f+1}")
    ax2.set_ylabel("cap-saturation rate"); ax2.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=7); ax2.legend(loc="lower right", fontsize=7)
    ax.set_title("Candidate-set size & 200-cap saturation vs layer"); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "setsize_vs_layer.png"), dpi=130); plt.close(fig)

    # ---- 4. heatmap: layer x position recall@K (masked), per forward ----
    ki = K_ABS.index(args.recallK) if args.recallK in K_ABS else 2
    fig, axes = plt.subplots(1, F, figsize=(4.2 * F, 4.2), squeeze=False)
    r = masked_recall_abs(ki)              # [NL,F,P]
    for f in range(F):
        ax = axes[0][f]
        im = ax.imshow(r[:, f, :], aspect="auto", origin="lower", vmin=0, vmax=1, cmap="magma")
        ax.set_title(f"F{f+1}  recall@{K_ABS[ki]}"); ax.set_xlabel("position");
        if f == 0:
            ax.set_ylabel("layer")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8)
    fig.suptitle(f"Masked recall@{K_ABS[ki]}: layer × position");
    fig.savefig(os.path.join(out_dir, "heatmap_recall.png"), dpi=130); plt.close(fig)

    # ---- 5 & 6. per-position bars, revealed vs masked, one panel per K (chosen layer, F1) ----
    def bar_grid(hit, Ks, klabels, fname, title):
        ncol = 3
        nrow = int(np.ceil(len(Ks) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3 * nrow), squeeze=False)
        xpos = np.arange(Pn)
        cm = count[best_layer, 0, :, MASKED]
        cr = count[best_layer, 0, :, REVEALED]
        for ki2, lab in enumerate(klabels):
            ax = axes[ki2 // ncol][ki2 % ncol]
            rm = _safe(hit[ki2, best_layer, 0, :, MASKED], cm)
            rr = _safe(hit[ki2, best_layer, 0, :, REVEALED], cr)
            ax.bar(xpos - 0.2, rm, width=0.4, label="masked")
            ax.bar(xpos + 0.2, rr, width=0.4, label="revealed")
            ax.set_title(lab); ax.set_ylim(0, 1); ax.set_xlabel("position")
            if ki2 == 0:
                ax.legend(fontsize=7)
        for j in range(len(Ks), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle(f"{title}  (layer {best_layer}, F1)"); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=130); plt.close(fig)

    bar_grid(d["hit_abs"], K_ABS, [f"recall@{K}" for K in K_ABS],
             "position_bars_abs.png", "Recall by position: revealed vs masked")
    bar_grid(d["hit_pct"], K_PCT, [f"top-{q*100:g}%" for q in K_PCT],
             "position_bars_pct.png", "Percentile-recall by position: revealed vs masked")

    # ---- 7. conditional (Markov) path: prefix-all-correct recall vs unconditional (chosen layer, F1) ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xpos = np.arange(Pn)
    cc = d["cond_count"][best_layer, 0, :]
    for ki2, K in enumerate(K_COND):
        cond = _safe(d["cond_hit"][ki2, best_layer, 0, :], cc)
        ax.plot(xpos, cond, marker="o", ms=3, label=f"cond@{K}")
    # unconditional masked recall@1 for contrast
    uncond1 = _safe(d["hit_abs"][0, best_layer, 0, :, MASKED], count[best_layer, 0, :, MASKED])
    ax.plot(xpos, uncond1, ls="--", c="gray", label="uncond@1 (masked)")
    ax.set_xlabel("position i"); ax.set_ylabel("recall"); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    ax.set_title(f"Conditional path: recall@K at i | positions 0..i-1 all gold-top-1  (layer {best_layer}, F1)")
    ax.legend(fontsize=7, ncol=3); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "conditional_markov.png"), dpi=130); plt.close(fig)

    print(f"[plot] best layer (by F1 nucleus recall) = {best_layer} "
          f"(recall={nuc_recall_L[best_layer,0]:.3f}, mean|S|={size_mean_L[best_layer,0]:.1f})")
    print(f"[plot] wrote 7 figures to {out_dir}")


if __name__ == "__main__":
    main()
