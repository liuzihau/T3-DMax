# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# LENS-SURFACE OFFLINE ANALYSIS -- consumes the npz shards written by probe_lens_surface.py and produces
# every figure/number in arc_head/exp_lens_surface.md trial 1: the layer x step and layer x delta surfaces,
# the masked-vs-clean decomposition, the teacher verdict (q_conv vs q_read), teacher softness, the
# stability score, the metric-5 oracle difficulty map, a qualitative cell dump, and REPORT.md + summary.json.
#
# Pure numpy/pandas/matplotlib; runs on a CPU box. No torch, no GPU.
#
# Run:
#   python probe_lens_analyze.py --run_dir runs/lens_surface [--out_dir ...] [--tokenizer_path <ckpt>]
#       [--sample_frac 1.0] [--qual_n 300]

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
try:   # decoded tokens/prompts contain '$'; without this matplotlib sends them to the TeX parser
    matplotlib.rcParams["text.parse_math"] = False
except KeyError:
    pass
import matplotlib.pyplot as plt  # noqa: E402

D_READING = 999
SCALAR_FIELDS = ["sample", "block", "d", "layer", "pos", "state", "commit_step", "delta",
                 "d_conv", "t_c", "q_tc", "rank_tc", "entropy", "kl_f", "kl_conv", "kl_read"]
# layer x step / layer x delta surface metrics: (column, aggregator-friendly name, colorbar label)
SURFACE_METRICS = [
    ("r1", "recall@1", "P(argmax == t_c)"),
    ("q_tc", "q(t_c)", "mean prob of t_c"),
    ("entropy", "entropy", "mean entropy (nats)"),
    ("kl_f", "KL(f||q)", "vs same-step final layer"),
    ("kl_conv", "KL(q_conv||q)", "vs same layer @ converge step"),
    ("kl_read", "KL(q_read||q)", "vs same layer @ reading fwd"),
]


def load_shards(run_dir, sample_frac=1.0, seed=0):
    paths = sorted(glob.glob(os.path.join(run_dir, "probe_lens_surface_shard*.npz")))
    if not paths:
        raise SystemExit(f"no shards found under {run_dir}")
    cols = {f: [] for f in SCALAR_FIELDS}
    ids_l, probs_l = [], []
    rng = np.random.default_rng(seed)
    for p in paths:
        z = np.load(p)
        n = z["sample"].shape[0]
        keep = None
        if sample_frac < 1.0:
            keep = rng.random(n) < sample_frac
        for f in SCALAR_FIELDS:
            a = z[f]
            cols[f].append(a[keep] if keep is not None else a)
        ids_l.append(z["top_ids"][keep] if keep is not None else z["top_ids"])
        probs_l.append(z["top_probs"][keep] if keep is not None else z["top_probs"])
    df = pd.DataFrame({f: np.concatenate(cols[f]) for f in SCALAR_FIELDS})
    top_ids = np.concatenate(ids_l)
    top_probs = np.concatenate(probs_l).astype(np.float32)
    print(f"[analyze] loaded {len(df):,} rows from {len(paths)} shards")
    return df, top_ids, top_probs


def add_derived(df, top_ids, top_probs, have_read):
    df["r1"] = (df["rank_tc"] == 0).astype(np.float32)
    df["r10"] = (df["rank_tc"] < 10).astype(np.float32)
    df["top1_id"] = top_ids[:, 0]
    df["top1_p"] = top_probs[:, 0]
    df["ncand"] = (top_probs > 0.01).sum(axis=1).astype(np.int8)
    df["is_read"] = df["d"].values == D_READING
    # entering step d, a position is mask input iff d <= its commit step  <=>  delta <= 0
    df["pre_commit"] = (~df["is_read"]) & (df["delta"] <= 0)
    if not have_read:
        df.loc[:, "kl_read"] = np.nan
    return df


def _heat(ax, piv, title, cbar_label, fig, vmax=None):
    """One layer x step/delta heatmap. viridis (perceptually uniform, CVD-safe); NaN cells blank."""
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad(color="#00000000")
    im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap=cmap, vmax=vmax,
                   extent=[piv.columns.min() - 0.5, piv.columns.max() + 0.5,
                           piv.index.min() - 0.5, piv.index.max() + 0.5])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(piv.columns.name, fontsize=9)
    ax.set_ylabel("layer (0 = embeddings)", fontsize=9)
    ax.tick_params(labelsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def fig_surfaces(df, out_dir, xcol, subset_name, subset_mask, fname, max_abs_x=None):
    """Grid of surface heatmaps: rows of SURFACE_METRICS over layer x <xcol>."""
    sub = df[subset_mask]
    if max_abs_x is not None:
        sub = sub[sub[xcol].abs() <= max_abs_x]
    if len(sub) == 0:
        print(f"[analyze] skip {fname}: empty subset")
        return
    metrics = [m for m in SURFACE_METRICS if not sub[m[0]].isna().all()]
    ncol = 3
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow), squeeze=False)
    for i, (col, name, cbl) in enumerate(metrics):
        piv = sub.pivot_table(index="layer", columns=xcol, values=col, aggfunc="mean")
        piv.columns.name = xcol
        _heat(axes[i // ncol][i % ncol], piv, name, cbl, fig)
    for j in range(len(metrics), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Lens surface -- {subset_name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)
    print(f"[analyze] wrote {fname}")


def fig_decomposition(df, out_dir):
    """Masked-vs-clean per-layer curves + the gap (the genuine-uncertainty component)."""
    masked = df[df["pre_commit"]].groupby("layer")[["r1", "q_tc"]].mean()
    clean = df[(~df["is_read"]) & (df["delta"] >= 1)].groupby("layer")[["r1", "q_tc"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, ttl in [(axes[0], "r1", "recall@1 vs t_c"), (axes[1], "q_tc", "mean q(t_c)")]:
        ax.plot(clean.index, clean[col], color="#1b6ca8", lw=2, label="clean input (reading own token)")
        ax.plot(masked.index, masked[col], color="#e07b39", lw=2, label="masked input (pre-commit)")
        gap = (clean[col] - masked[col]).reindex(clean.index)
        ax.plot(gap.index, gap, color="#6a6a6a", lw=1.5, ls="--", label="gap = genuine uncertainty")
        ax.set_xlabel("layer (0 = embeddings)")
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Clean-band decomposition: workspace/rotation effect vs genuine uncertainty", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "fig3_decomposition.png"), dpi=150)
    plt.close(fig)
    print("[analyze] wrote fig3_decomposition.png")
    return masked, clean


def teacher_verdict(df, out_dir, have_read):
    """q_conv vs q_read per cell: kl_read at d == d_conv, split by whether the position was committed at
    the final decode step (mask input at the conv forward) vs earlier. Plus top-1 agreement via join."""
    if not have_read:
        print("[analyze] reading forward disabled -- skipping teacher verdict")
        return None
    conv = df[(~df["is_read"]) & (df["d"] == df["d_conv"])].copy()
    conv["last_step_commit"] = conv["commit_step"] == conv["d_conv"]
    read = df[df["is_read"]][["sample", "block", "layer", "pos", "top1_id"]].rename(
        columns={"top1_id": "top1_read"})
    j = conv.merge(read, on=["sample", "block", "layer", "pos"], how="inner")
    j["top1_agree"] = (j["top1_id"] == j["top1_read"]).astype(np.float32)

    g_kl = j.groupby(["layer", "last_step_commit"])["kl_read"].mean().unstack()
    g_ag = j.groupby(["layer", "last_step_commit"])["top1_agree"].mean().unstack()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, g, ttl, yl in [(axes[0], g_kl, "KL(q_read || q_conv) per layer", "mean KL (nats)"),
                           (axes[1], g_ag, "top-1 agreement conv vs read", "fraction agree")]:
        if False in g.columns:
            ax.plot(g.index, g[False], color="#1b6ca8", lw=2, label="committed earlier (clean at conv fwd)")
        if True in g.columns:
            ax.plot(g.index, g[True], color="#e07b39", lw=2, label="committed at final step (mask at conv fwd)")
        ax.set_xlabel("layer (0 = embeddings)")
        ax.set_ylabel(yl, fontsize=9)
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Teacher verdict: can the free last forward replace the clean reading forward?", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(out_dir, "fig4_teacher_verdict.png"), dpi=150)
    plt.close(fig)
    print("[analyze] wrote fig4_teacher_verdict.png")
    out = {
        "top1_agreement_overall": float(j["top1_agree"].mean()),
        "top1_agreement_last_step_commit": float(j.loc[j["last_step_commit"], "top1_agree"].mean())
        if j["last_step_commit"].any() else None,
        "top1_agreement_earlier_commit": float(j.loc[~j["last_step_commit"], "top1_agree"].mean())
        if (~j["last_step_commit"]).any() else None,
        "kl_read_at_conv_mean": float(j["kl_read"].mean()),
        "frac_cells_last_step_commit": float(j["last_step_commit"].mean()),
    }
    return out


def fig_softness(df, out_dir, have_read):
    """Teacher softness at the reading forward (fallback: converge step) per layer."""
    src = df[df["is_read"]] if have_read else df[(~df["is_read"]) & (df["d"] == df["d_conv"])]
    label = "reading forward" if have_read else "converge step (no reading fwd)"
    g = src.groupby("layer")[["entropy", "top1_p", "ncand"]].mean()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, col, ttl in [(axes[0], "entropy", "mean entropy (nats)"),
                         (axes[1], "top1_p", "mean top-1 prob"),
                         (axes[2], "ncand", "mean #candidates > 1%")]:
        ax.plot(g.index, g[col], color="#1b6ca8", lw=2)
        ax.set_xlabel("layer (0 = embeddings)")
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
    fig.suptitle(f"Teacher softness per layer ({label}) -- decides L1 semantics (CE-like vs candidate-set)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(out_dir, "fig5_softness.png"), dpi=150)
    plt.close(fig)
    print("[analyze] wrote fig5_softness.png")
    return g


def stability(df, top_ids, out_dir, chunk=2_000_000):
    """Top-10 overlap between adjacent denoise steps at the same (sample, block, pos, layer), masked
    pre-commit cells only. Persistent alternatives = workspace-like; churn = noise."""
    sub = df[df["pre_commit"]]
    idx = sub.sort_values(["sample", "block", "pos", "layer", "d"], kind="mergesort").index.to_numpy()
    if idx.size < 2:
        print("[analyze] stability: not enough cells")
        return None
    s = df.loc[idx]
    key_same = ((s["sample"].values[:-1] == s["sample"].values[1:]) &
                (s["block"].values[:-1] == s["block"].values[1:]) &
                (s["pos"].values[:-1] == s["pos"].values[1:]) &
                (s["layer"].values[:-1] == s["layer"].values[1:]) &
                (s["d"].values[1:] == s["d"].values[:-1] + 1))
    a_idx, b_idx = idx[:-1][key_same], idx[1:][key_same]
    if a_idx.size == 0:
        print("[analyze] stability: no adjacent-step pairs")
        return None
    ov = np.empty(a_idx.size, dtype=np.float32)
    for lo in range(0, a_idx.size, chunk):
        hi = min(lo + chunk, a_idx.size)
        A, B = top_ids[a_idx[lo:hi]], top_ids[b_idx[lo:hi]]
        ov[lo:hi] = (A[:, :, None] == B[:, None, :]).any(-1).sum(-1) / A.shape[1]
    sdf = pd.DataFrame({"layer": df.loc[a_idx, "layer"].values,
                        "delta": df.loc[a_idx, "delta"].values, "overlap": ov})
    piv = sdf.pivot_table(index="layer", columns="delta", values="overlap", aggfunc="mean")
    piv = piv[[c for c in piv.columns if -12 <= c <= 0]]
    piv.columns.name = "delta (steps before commit)"
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _heat(ax, piv, "top-10 stability across adjacent steps (masked cells)", "mean overlap", fig, vmax=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig6_stability.png"), dpi=150)
    plt.close(fig)
    print("[analyze] wrote fig6_stability.png")
    return {"stability_mean": float(ov.mean()),
            "stability_by_layer": sdf.groupby("layer")["overlap"].mean().round(4).to_dict()}


def fig_difficulty(df, out_dir, have_read):
    """Metric-5 oracle difficulty map: at the FIRST forward (d=0), how far is each (layer, pos) readout
    from its own converged/reading view -- the gap ARC's branch must close."""
    col = "kl_read" if have_read else "kl_conv"
    sub = df[(df["d"] == 0) & (~df["is_read"])]
    if sub[col].isna().all():
        print("[analyze] difficulty map: no reference available")
        return
    piv = sub.pivot_table(index="layer", columns="pos", values=col, aggfunc="mean")
    piv.columns.name = "block position"
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _heat(ax, piv, f"ARC oracle difficulty map: {col} at first forward", "mean KL (nats)", fig)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig7_difficulty_map.png"), dpi=150)
    plt.close(fig)
    print("[analyze] wrote fig7_difficulty_map.png")


def qualitative_dump(df, top_ids, top_probs, out_dir, qual_n, tokenizer_path=None, seed=0):
    """Mid-layer disagreeing masked cells for human reading (workspace hunt)."""
    NL = int(df["layer"].max()) + 1
    lo, hi = NL // 3, 2 * NL // 3
    sub = df[(df["pre_commit"]) & (df["layer"].between(lo, hi)) & (df["rank_tc"] >= 10)]
    if len(sub) == 0:
        print("[analyze] qualitative: no matching cells")
        return None
    take = sub.sample(n=min(qual_n, len(sub)), random_state=seed).sort_values(["layer", "delta"])
    rows = take[["sample", "block", "d", "layer", "pos", "delta", "t_c", "rank_tc", "q_tc"]].copy()
    ids = top_ids[take.index.to_numpy()]
    probs = top_probs[take.index.to_numpy()]
    tok = None
    if tokenizer_path:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            print(f"[analyze] tokenizer unavailable ({e}); dumping raw ids")
    if tok is not None:
        rows["t_c_str"] = [tok.decode([int(t)]) for t in rows["t_c"]]
        rows["top10"] = [" | ".join(f"{tok.decode([int(i)])}:{p:.2f}" for i, p in zip(r_i, r_p))
                         for r_i, r_p in zip(ids, probs)]
    else:
        rows["top10"] = [" | ".join(f"{int(i)}:{p:.2f}" for i, p in zip(r_i, r_p))
                         for r_i, r_p in zip(ids, probs)]
    path = os.path.join(out_dir, "qualitative_cells.csv")
    rows.to_csv(path, index=False)
    print(f"[analyze] wrote {path} ({len(rows)} cells)")
    return path


def crystallization(df):
    """Per-layer masked pre-commit recall curves + the x* estimate (0.8 x top-layer recall@10)."""
    g = df[df["pre_commit"]].groupby("layer")[["r1", "r10"]].mean()
    top = g["r10"].iloc[-1]
    ok = g[g["r10"] >= 0.8 * top]
    x_star = int(ok.index[0]) if len(ok) else None
    return g, x_star, float(top)


def main():
    p = argparse.ArgumentParser(description="Offline analysis for the lens-surface probe (trial 1)")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--sample_frac", type=float, default=1.0, help="row subsample for quick looks")
    p.add_argument("--qual_n", type=int, default=300)
    args = p.parse_args()
    out_dir = args.out_dir or os.path.join(args.run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    meta_path = os.path.join(args.run_dir, "probe_lens_surface_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    have_read = bool(meta.get("reading_forward", True))

    df, top_ids, top_probs = load_shards(args.run_dir, args.sample_frac)
    df = add_derived(df, top_ids, top_probs, have_read)

    # F1/F2: surfaces (standard + event-aligned frames), masked and clean separately where meaningful
    fig_surfaces(df, out_dir, "d", "masked input cells (standard frame)",
                 (~df["is_read"]) & (df["state"] == 0), "fig1_surface_masked.png")
    fig_surfaces(df, out_dir, "d", "clean input cells (standard frame)",
                 (~df["is_read"]) & (df["state"] == 1), "fig1_surface_clean.png")
    fig_surfaces(df, out_dir, "delta", "event-aligned frame (delta = d - commit step)",
                 ~df["is_read"], "fig2_event_aligned.png", max_abs_x=12)

    masked_g, clean_g = fig_decomposition(df, out_dir)
    verdict = teacher_verdict(df, out_dir, have_read)
    soft_g = fig_softness(df, out_dir, have_read)
    stab = stability(df, top_ids, out_dir)
    fig_difficulty(df, out_dir, have_read)
    qual_path = qualitative_dump(df, top_ids, top_probs, out_dir, args.qual_n, args.tokenizer_path)
    cryst_g, x_star, top_r10 = crystallization(df)

    summary = {
        "rows": int(len(df)),
        "meta": meta,
        "crystallization_layer_x_star": x_star,
        "crystallization_rule": "min layer with masked pre-commit recall@10 >= 0.8 x top-layer",
        "top_layer_masked_recall10": top_r10,
        "masked_recall1_by_layer": cryst_g["r1"].round(4).to_dict(),
        "teacher_verdict": verdict,
        "softness_by_layer": soft_g.round(4).to_dict() if soft_g is not None else None,
        "stability": stab,
        "qualitative_csv": qual_path,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    lines = [
        "# Lens-surface trial-1 analysis", "",
        f"Rows analyzed: {len(df):,} (sample_frac={args.sample_frac})", "",
        f"**Crystallization layer x\\*** (masked pre-commit recall@10 reaches 80% of top layer): "
        f"**{x_star}** (top-layer masked recall@10 = {top_r10:.3f})", "",
    ]
    if verdict:
        lines += [
            "**Teacher verdict (q_conv vs q_read):** "
            f"top-1 agreement overall = {verdict['top1_agreement_overall']:.3f}; "
            f"last-step-committed cells = {verdict['top1_agreement_last_step_commit']:.3f}; "
            f"earlier-committed cells = {verdict['top1_agreement_earlier_commit']:.3f}; "
            f"fraction last-step-committed = {verdict['frac_cells_last_step_commit']:.3f}. "
            "High agreement everywhere => the free last forward can be ARC's teacher; "
            "divergence concentrated in last-step-committed cells => use the reading forward.", "",
        ]
    if stab:
        lines += [f"**Stability (masked cells, adjacent-step top-10 overlap):** mean = "
                  f"{stab['stability_mean']:.3f}. High stability + low agreement at mid layers = "
                  "workspace-like; low stability = noise (see fig6 + qualitative_cells.csv).", ""]
    lines += ["Figures: fig1 (surfaces, masked/clean), fig2 (event-aligned), fig3 (decomposition), "
              "fig4 (teacher verdict), fig5 (softness), fig6 (stability), fig7 (ARC difficulty map).",
              "", "Decision table: see arc_head/exp_lens_surface.md section 7."]
    with open(os.path.join(out_dir, "REPORT.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"[analyze] wrote summary.json + REPORT.md -> {out_dir}")


if __name__ == "__main__":
    main()
