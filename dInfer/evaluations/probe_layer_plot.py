# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# Plot the layer-readout probe (probe_layer_readout.py) -- POSITION-RESOLVED throughout. DMax commits
# left-to-right, so recall is a strong function of position-in-block; NOTHING here averages positions away
# except the two explicitly-labelled per-layer summaries. Recall@K (accuracy) and |S| (sharpness) are kept
# separate -- the nucleus(0.5) number conflates them (late layers are overconfident-but-wrong, early layers
# are flat). Everything below is FORWARD 1 (block all-masked, the hard case) unless a figure says otherwise.
#
# Figures:
#   1. headline_f1.png        -- layer x position heatmaps, F1: recall@K | nucleus-recall | mean |S|.
#   2. heatmap_recallK_byfwd  -- layer x position recall@K, one panel per forward (the F1->F2->F3 progression).
#   3. recall_vs_position     -- F1: x=position, y=recall@K, one line per layer (how far into the block recall reaches).
#   4. decision_recallK       -- F1: per selected position, recall@K vs K (log), one line per layer. THE decision:
#                                at an early position, is there a mid-layer where recall~1 at small K?
#   5. position_bars_abs/pct  -- best layer, F1: recall by position, masked vs revealed bars.
#   6. conditional_markov     -- prefix-all-correct recall@K vs position (does a correct left context help?).
#
# Run:  python probe_layer_plot.py --npz runs/probe_readout.npz [--recallK 10]

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MASKED, REVEALED = 0, 1


def _safe(a, b):
    return np.divide(a, b, out=np.full_like(a, np.nan, dtype=np.float64), where=b > 0)


def _layer_subset(NL, k=6):
    if NL <= k:
        return list(range(NL))
    s = sorted(set(int(round(x)) for x in np.linspace(0, NL - 1, k)))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--recallK", type=int, default=10, help="K for recall@K heatmaps/lines")
    ap.add_argument("--positions", default="0,1,2,4,8,16", help="positions = decision_recallK facets")
    ap.add_argument("--layers", default="0,8,12,14,16,18,20",
                    help="layers shown as LINES in recall_vs_position & decision_recallK (the candidate "
                         "insertion depths). Out-of-range entries are dropped; empty -> auto ~6.")
    args = ap.parse_args()

    d = np.load(args.npz)
    meta = json.load(open(os.path.splitext(args.npz)[0] + "_meta.json"))
    K_ABS, K_PCT, K_COND = meta["K_ABS"], meta["K_PCT"], meta["K_COND"]
    NL, F, Pn = d["count"].shape[:3]
    ki = K_ABS.index(args.recallK) if args.recallK in K_ABS else 2
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.npz)), "plots")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[plot] layers={NL} forwards={F} positions={Pn} agree={meta.get('agree'):.3f} "
          f"recallK={K_ABS[ki]} -> {out_dir}")

    count = d["count"]                                  # [NL,F,P,2]

    def rec(hit_ki, f, state=MASKED):                  # recall@K map [NL,P] for forward f
        return _safe(d["hit_abs"][hit_ki, :, f, :, state], count[:, f, :, state])

    def nuc(f, state=MASKED):
        return _safe(d["hit_nuc"][:, f, :, state], count[:, f, :, state])

    def size(f, state=MASKED):
        return _safe(d["size_sum"][:, f, :, state], count[:, f, :, state])

    def _hm(ax, M, title, vmin, vmax, cmap, cbar_label):
        cm = plt.get_cmap(cmap).copy(); cm.set_bad("lightgray")     # gray = no masked data at that (layer,pos)
        im = ax.imshow(M, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap=cm)
        ax.set_title(title); ax.set_xlabel("position in block"); ax.set_ylabel("layer")
        plt.colorbar(im, ax=ax, shrink=0.85, label=cbar_label)
        return im

    # best layer = the one with highest masked recall@K averaged over the EARLY positions (0..3), F1
    early = slice(0, min(4, Pn))
    early_rec = np.nanmean(rec(ki, 0)[:, early], axis=1)
    best_layer = int(np.nanargmax(np.where(np.isnan(early_rec), -1, early_rec)))

    # ---- 1. headline F1: recall@K | nucleus-recall | mean|S|, all layer x position ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    _hm(ax[0], rec(ki, 0), f"masked recall@{K_ABS[ki]}  (F1)", 0, 1, "magma", "recall")
    _hm(ax[1], nuc(0), "masked nucleus(0.5)∩200 recall  (F1)", 0, 1, "magma", "recall")
    _hm(ax[2], size(0), "mean |S| = nucleus∩200 size  (F1)", 0, meta["nucleus_cap"], "viridis", "|S|")
    fig.suptitle("HEADLINE (Forward 1): accuracy (left/mid) vs sharpness (right), by layer × position")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "headline_f1.png"), dpi=130); plt.close(fig)

    # ---- 2. recall@K heatmap per forward ----
    fig, ax = plt.subplots(1, F, figsize=(5.2 * F, 4.4), squeeze=False)
    for f in range(F):
        _hm(ax[0][f], rec(ki, f), f"F{f+1}  masked recall@{K_ABS[ki]}", 0, 1, "magma", "recall")
    fig.suptitle(f"Masked recall@{K_ABS[ki]} across forwards (F1=all-masked → F3=most revealed)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "heatmap_recallK_byfwd.png"), dpi=130); plt.close(fig)

    # ---- 3. recall@K vs position, one line per layer (F1) ----
    fig, axp = plt.subplots(figsize=(9, 5))
    r = rec(ki, 0)                                     # [NL,P]
    subs = [int(x) for x in args.layers.split(",") if x.strip() != "" and 0 <= int(x) < NL] \
        if args.layers.strip() else _layer_subset(NL)
    if not subs:
        subs = _layer_subset(NL)
    cmap = plt.get_cmap("viridis")
    for L in subs:
        axp.plot(np.arange(Pn), r[L], marker="o", ms=3, color=cmap(L / max(NL - 1, 1)), label=f"L{L}")
    axp.set_xlabel("position in block"); axp.set_ylabel(f"masked recall@{K_ABS[ki]}"); axp.set_ylim(0, 1)
    axp.grid(alpha=0.3); axp.legend(fontsize=7, ncol=2, title="layer")
    axp.set_title(f"recall@{K_ABS[ki]} vs position (F1). NOTE: F1 = block all-masked + bidirectional attn,\n"
                  "so positions are ~symmetric → FLAT is expected here; the left→right decay appears at F2/F3")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "recall_vs_position.png"), dpi=130); plt.close(fig)

    # ---- 4. THE decision: recall@K vs K (log), per selected position, line per layer (F1) ----
    pos_list = [int(p) for p in args.positions.split(",") if int(p) < Pn]
    ncol = 3; nrow = int(np.ceil(len(pos_list) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.6 * nrow), squeeze=False)
    for pi, p in enumerate(pos_list):
        ax = axes[pi // ncol][pi % ncol]
        for L in subs:
            ys = [_safe(d["hit_abs"][k, L, 0, p, MASKED], count[L, 0, p, MASKED]) for k in range(len(K_ABS))]
            ax.plot(K_ABS, ys, marker="o", ms=3, color=cmap(L / max(NL - 1, 1)), label=f"L{L}")
        ax.set_xscale("log"); ax.set_xticks(K_ABS); ax.set_xticklabels(K_ABS)
        ax.axhline(0.99, ls="--", c="r", lw=0.8)
        n0 = count[best_layer, 0, p, MASKED]
        ax.set_title(f"position {p}  (n={int(n0)} masked@F1)")
        ax.set_xlabel("candidate-set size K (top-K)"); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3, which="both")
        if pi == 0:
            ax.set_ylabel("masked recall@K"); ax.legend(fontsize=6, ncol=2, title="layer")
    for j in range(len(pos_list), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("DECISION: at each position, how big must top-K be to capture gold? (F1)\n"
                 "GO if an EARLY position has a mid-layer curve reaching ~1 at small K")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "decision_recallK.png"), dpi=130); plt.close(fig)

    # ---- 5. per-position bars, revealed vs masked (best layer, F1) ----
    def bar_grid(hit, Ks, klabels, fbase, title, f):
        # F1 has NO revealed positions (block all-masked) -> orange bars appear only at F2/F3, where the
        # left prefix is committed. Revealed = ceiling/sanity (input IS the token); masked = the real test.
        ncol = 3; nrow = int(np.ceil(len(Ks) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3 * nrow), squeeze=False)
        xpos = np.arange(Pn); cm = count[best_layer, f, :, MASKED]; cr = count[best_layer, f, :, REVEALED]
        for k2, lab in enumerate(klabels):
            ax = axes[k2 // ncol][k2 % ncol]
            ax.bar(xpos - 0.2, np.nan_to_num(_safe(hit[k2, best_layer, f, :, MASKED], cm)), 0.4, label="masked")
            ax.bar(xpos + 0.2, np.nan_to_num(_safe(hit[k2, best_layer, f, :, REVEALED], cr)), 0.4, label="revealed")
            ax.set_title(lab); ax.set_ylim(0, 1); ax.set_xlabel("position")
            if k2 == 0:
                ax.legend(fontsize=7)
        for j in range(len(Ks), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle(f"{title}  (layer {best_layer}, F{f+1})"); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{fbase}_f{f+1}.png"), dpi=130); plt.close(fig)

    for f in range(F):                                  # one grid per forward -> revealed bars show at F2/F3
        bar_grid(d["hit_abs"], K_ABS, [f"recall@{K}" for K in K_ABS],
                 "position_bars_abs", "Recall by position: revealed (ceiling) vs masked (real test)", f)
        bar_grid(d["hit_pct"], K_PCT, [f"top-{q*100:g}%" for q in K_PCT],
                 "position_bars_pct", "Percentile-recall by position: revealed vs masked", f)

    # ---- 6. conditional (Markov) path: prefix-all-correct recall@K vs position (best layer, F1) ----
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xpos = np.arange(Pn); cc = d["cond_count"][best_layer, 0, :]
    for k2, K in enumerate(K_COND):
        ax.plot(xpos, _safe(d["cond_hit"][k2, best_layer, 0, :], cc), marker="o", ms=3, label=f"cond@{K}")
    ax.plot(xpos, _safe(d["hit_abs"][0, best_layer, 0, :, MASKED], count[best_layer, 0, :, MASKED]),
            ls="--", c="gray", label="uncond@1 (masked)")
    ax.set_xlabel("position i"); ax.set_ylabel("recall"); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    ax.set_title(f"Conditional path: recall@K at i | positions 0..i-1 all gold-top-1  (layer {best_layer}, F1)")
    ax.legend(fontsize=7, ncol=3); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "conditional_markov.png"), dpi=130); plt.close(fig)

    print(f"[plot] best early-position layer (recall@{K_ABS[ki]}, pos0-3, F1) = {best_layer}")
    print(f"[plot] wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
