# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# AGGREGATE POSITION x LAYER MAPS -- the population version of the per-block dashboard.
#
# Same axes as the dashboard (y = block position, 0 at the top; x = layer, first -> last), but every
# cell aggregates over ALL samples and blocks in the run instead of showing one block's tokens.
# Cell value = KL from the layer's own CONVERGED distribution (kl_conv by default; --ref read uses the
# clean-token reading forward instead). Denoise steps are pooled, not separated.
#
#   fig1  median KL          -- left = masked-input cells, right = clean-input cells
#   fig2  95th-percentile KL -- same two panels (the tail: how bad do the worst cells get?)
#   fig3  median KL at three points around each position's own commit step:
#           left = d-1 (still masked), middle = d (the forward that commits it), right = d+1 (now clean)
#
# Panels within a figure share one colour scale, so they are directly comparable.
# Pure numpy/pandas/matplotlib; CPU only.
#
# Run:
#   python probe_lens_aggmap.py --run_dir runs/lens_surface
#   # options: --ref conv|read  --stat median|p95 (fig3)  --layers 1:  --pos_range 0:32  --min_count 5

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
try:   # decoded text elsewhere in this toolchain contains '$'; keep matplotlib out of mathtext mode
    matplotlib.rcParams["text.parse_math"] = False
except KeyError:
    pass
import matplotlib.pyplot as plt  # noqa: E402

D_READING = 999
NEED = ["layer", "pos", "state", "delta", "d", "kl_conv", "kl_read"]


def load(run_dir, ref, include_reading):
    paths = sorted(glob.glob(os.path.join(run_dir, "probe_lens_surface_shard*.npz")))
    if not paths:
        raise SystemExit(f"no shards under {run_dir}")
    cols = {k: [] for k in NEED}
    for p in paths:
        z = np.load(p)
        for k in NEED:
            cols[k].append(z[k] if k in z.files else np.full(z["layer"].shape, np.nan, np.float32))
    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    # the shards store fp16; pandas' groupby median/quantile has no fp16 kernel
    df["kl"] = (df["kl_read"] if ref == "read" else df["kl_conv"]).astype(np.float32)
    if ref == "read" and (df["kl"] < 0).all():
        raise SystemExit("--ref read requested but this run has no reading forward (kl_read = -1)")
    df = df[df["kl"] >= 0]                                   # -1 marks 'not recorded'
    if not include_reading:
        df = df[df["d"] != D_READING]                        # the extra clean forward is not a decode step
    print(f"[agg] {len(df):,} cells from {len(paths)} shards (ref={ref})")
    return df


def grid(sub, stat, positions, layers, min_count):
    """(position x layer) grid of the chosen statistic; cells with too few samples -> NaN."""
    if len(sub) == 0:
        return np.full((len(positions), len(layers)), np.nan), np.zeros((len(positions), len(layers)))
    agg = sub.groupby(["pos", "layer"])["kl"]
    val = (agg.median() if stat == "median" else agg.quantile(0.95)).unstack()
    cnt = agg.size().unstack()
    val = val.reindex(index=positions, columns=layers)
    cnt = cnt.reindex(index=positions, columns=layers).fillna(0)
    v = val.to_numpy(dtype=float)
    c = cnt.to_numpy(dtype=float)
    v[c < min_count] = np.nan
    return v, c


def draw(fig, ax, v, positions, layers, title, norm, cmap, show_y, fs):
    im = ax.imshow(v, aspect="auto", origin="upper", cmap=cmap, norm=norm,
                   extent=[-0.5, len(layers) - 0.5, len(positions) - 0.5, -0.5])
    ax.set_title(title, fontsize=fs + 2, fontweight="bold", pad=8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers], fontsize=fs - 2)
    ax.set_xlabel("layer  (first -> last)", fontsize=fs)
    if show_y:
        ax.set_yticks(range(len(positions)))
        ax.set_yticklabels([str(p) for p in positions], fontsize=fs - 2)
        ax.set_ylabel("block position", fontsize=fs)
    else:
        ax.set_yticks([])
    ax.tick_params(length=0)
    return im


def figure(panels, positions, layers, suptitle, cbar_label, out_path, dpi, fs=11):
    """panels: list of (title, grid). One shared colour scale across the panels."""
    finite = np.concatenate([g[np.isfinite(g)].ravel() for _, g in panels if np.isfinite(g).any()])
    if finite.size == 0:
        print(f"[agg] skip {os.path.basename(out_path)}: no data")
        return
    vmax = float(np.percentile(finite, 99))
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=max(vmax, 1e-6))
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#eeeeee")                                  # under-sampled cells

    n = len(panels)
    w = 1.4 + n * (0.30 * len(layers) + 0.9)
    h = 1.9 + 0.26 * len(positions)
    fig, axes = plt.subplots(1, n, figsize=(w, h), squeeze=False)
    im = None
    for k, (title, g) in enumerate(panels):
        im = draw(fig, axes[0][k], g, positions, layers, title, norm, cmap, k == 0, fs)
    fig.suptitle(suptitle, fontsize=fs + 4, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.93, 0.96])
    cax = fig.add_axes([0.945, 0.12, 0.012, 0.72])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(cbar_label, fontsize=fs)
    cb.ax.tick_params(labelsize=fs - 2)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"[agg] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Aggregate position x layer KL maps")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--ref", default="conv", choices=["conv", "read"],
                    help="converged reference: 'conv' = same layer at the converge step (default), "
                         "'read' = same layer at the clean-token reading forward")
    ap.add_argument("--stat", default="median", choices=["median", "p95"], help="statistic for fig3")
    ap.add_argument("--layers", default="1:", help="layer slice, e.g. 1: (skip embeddings) or 10:21")
    ap.add_argument("--pos_range", default="all")
    ap.add_argument("--min_count", type=int, default=5, help="grey out cells with fewer samples")
    ap.add_argument("--include_reading", action="store_true",
                    help="also count the clean-token reading forward as a clean cell")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    df = load(args.run_dir, args.ref, args.include_reading)

    def rng(spec, lo, hi):
        if not spec or spec == "all":
            return lo, hi
        a, _, b = spec.partition(":")
        return (int(a) if a else lo), (int(b) if b else hi)

    lo_l, hi_l = rng(args.layers, int(df["layer"].min()), int(df["layer"].max()) + 1)
    layers = np.arange(max(lo_l, int(df["layer"].min())), min(hi_l, int(df["layer"].max()) + 1))
    all_pos = np.unique(df["pos"])
    lo_p, hi_p = rng(args.pos_range, int(all_pos.min()), int(all_pos.max()) + 1)
    positions = all_pos[(all_pos >= lo_p) & (all_pos < hi_p)]
    df = df[df["layer"].isin(layers) & df["pos"].isin(positions)]
    ref_txt = "converge step" if args.ref == "conv" else "reading forward"
    cbar = f"KL( q@{ref_txt} || q )  [nats]"

    masked, clean = df[df["state"] == 0], df[df["state"] == 1]
    print(f"[agg] masked cells {len(masked):,} | clean cells {len(clean):,}")

    # fig 1 -- median
    g_m, _ = grid(masked, "median", positions, layers, args.min_count)
    g_c, _ = grid(clean, "median", positions, layers, args.min_count)
    figure([("masked input", g_m), ("clean input", g_c)], positions, layers,
           f"Median KL to the {ref_txt} distribution  (all denoise steps pooled)",
           cbar, os.path.join(out_dir, "agg_fig1_median.png"), args.dpi)

    # fig 2 -- 95th percentile (worst tail)
    p_m, _ = grid(masked, "p95", positions, layers, args.min_count)
    p_c, _ = grid(clean, "p95", positions, layers, args.min_count)
    figure([("masked input", p_m), ("clean input", p_c)], positions, layers,
           f"95th-percentile KL to the {ref_txt} distribution  (worst-case cells)",
           cbar, os.path.join(out_dir, "agg_fig2_p95.png"), args.dpi)

    # fig 3 -- around each position's own commit step
    panels = []
    for dd, name in [(-1, "d-1  (still masked)"), (0, "d  (commits here)"), (1, "d+1  (now clean)")]:
        g, _ = grid(df[df["delta"] == dd], args.stat, positions, layers, args.min_count)
        panels.append((name, g))
    figure(panels, positions, layers,
           f"{'Median' if args.stat == 'median' else '95th-percentile'} KL around the commit step "
           f"(delta = d - commit step)",
           cbar, os.path.join(out_dir, f"agg_fig3_commit_{args.stat}.png"), args.dpi)

    meta = dict(run_dir=args.run_dir, ref=args.ref, stat_fig3=args.stat, min_count=args.min_count,
                include_reading=args.include_reading, n_cells=int(len(df)),
                n_masked=int(len(masked)), n_clean=int(len(clean)),
                layers=[int(x) for x in layers], positions=[int(x) for x in positions])
    with open(os.path.join(out_dir, "agg_maps_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[agg] done -> {out_dir}")


if __name__ == "__main__":
    main()
