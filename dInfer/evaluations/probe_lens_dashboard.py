# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# PER-BLOCK LENS DASHBOARD -- the zoom-in view of one (sample, block) from the lens-surface shards.
#
# Layout
#   row 1 : the prompt + the generated prefix leading into this block (plain text)
#   row 2 : one panel per selected denoise step (default 1, 3, 5, last). Each panel is
#             top strip   = the CONVERGED token t_c per block position
#             main grid   = layer (y) x block position (x); cell colour = divergence to t_c,
#                           cell text = that layer's top-k readout tokens (the converged token
#                           is [bracketed] when it appears)
#             bottom strip= the token actually FED at that position entering that step ([MASK] if masked)
#
# Everything comes from the npz shards written by probe_lens_surface.py; no GPU, no model.
# `in_tok` and the context json are newer fields -- older runs degrade gracefully (the bottom strip is
# then inferred from `state`, and the prompt row says the context file is missing).
#
# Run:
#   python probe_lens_dashboard.py --run_dir runs/lens_surface --sample 5 --block 3 \
#       --tokenizer_path /path/to/merged_ckpt
#   # options: --steps 1,3,5,-1 (1-based; -1 = last decode forward; R = the clean-token reading forward)
#   #          --layers 1:21  --metric nlp_tc|rank_tc|kl_conv|kl_read|entropy  --topk 5  --pos_range 0:32

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402

D_READING = 999
MASK_ID_DEFAULT = 156895
# CJK-capable families, in preference order. This vocab contains Chinese tokens; without one of these
# matplotlib draws tofu boxes and warns "Glyph ... missing from current font".
CJK_FONTS = ["Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Sans CJK TC", "Noto Sans SC", "Noto Sans JP",
             "Source Han Sans SC", "Source Han Sans", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
             "Droid Sans Fallback", "Arial Unicode MS", "Microsoft YaHei", "SimHei", "PingFang SC"]


def setup_font(user_font=None):
    """Append a CJK-capable family to the font fallback chains (matplotlib >= 3.6 falls back per glyph,
    so Latin text keeps its usual look and only missing glyphs come from the CJK font)."""
    from matplotlib import font_manager
    avail = {f.name for f in font_manager.fontManager.ttflist}
    picks = ([user_font] if user_font else []) + [f for f in CJK_FONTS if f in avail]
    picks = [p for p in picks if p]
    if picks:
        for key in ("font.sans-serif", "font.monospace"):
            matplotlib.rcParams[key] = list(matplotlib.rcParams[key]) + picks[:2]
        print(f"[dash] CJK fallback font: {picks[0]}")
        return True
    print("[dash] WARNING: no CJK-capable font installed -- Chinese tokens will render as boxes.\n"
          "        fix (pick one, then re-run):\n"
          "          pip install mplfonts && mplfonts init          # no sudo\n"
          "          sudo apt-get install -y fonts-noto-cjk         # system-wide\n"
          "        then clear the font cache:  rm -rf ~/.cache/matplotlib")
    return False
SCALARS = ["sample", "block", "d", "layer", "pos", "state", "commit_step", "delta",
           "d_conv", "t_c", "in_tok", "q_tc", "rank_tc", "entropy", "kl_f", "kl_conv", "kl_read"]
METRICS = {   # name -> (label, transform(df) -> value, vmax or None for auto)
    "nlp_tc":  ("-log q(t_c)  [0 = says the converged token]", lambda d: -np.log(np.clip(d["q_tc"], 1e-6, 1)), 12.0),
    "rank_tc": ("rank of t_c (log10)", lambda d: np.log10(1.0 + d["rank_tc"]), None),
    "kl_conv": ("KL(q_conv || q)", lambda d: d["kl_conv"], None),
    "kl_read": ("KL(q_read || q)", lambda d: d["kl_read"], None),
    "entropy": ("entropy (nats)", lambda d: d["entropy"], None),
}


def load_block(run_dir, sample, block):
    """Scan shards, keep only rows of this (sample, block). Returns dict of arrays + top-k arrays."""
    paths = sorted(glob.glob(os.path.join(run_dir, "probe_lens_surface_shard*.npz")))
    if not paths:
        raise SystemExit(f"no shards under {run_dir}")
    cols, ids_l, probs_l = {k: [] for k in SCALARS}, [], []
    have_in_tok = True
    for p in paths:
        z = np.load(p)
        have_in_tok = "in_tok" in z.files
        m = (z["sample"] == sample) & (z["block"] == block)
        if not m.any():
            continue
        for k in SCALARS:
            if k == "in_tok" and not have_in_tok:
                cols[k].append(np.full(int(m.sum()), -1, dtype=np.int32))
            else:
                cols[k].append(z[k][m])
        ids_l.append(z["top_ids"][m])
        probs_l.append(z["top_probs"][m])
    if not ids_l:
        raise SystemExit(f"sample={sample} block={block} not found in {run_dir}")
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["top_ids"] = np.concatenate(ids_l)
    out["top_probs"] = np.concatenate(probs_l).astype(np.float32)
    out["_have_in_tok"] = have_in_tok
    return out


def load_context(run_dir, sample):
    for p in sorted(glob.glob(os.path.join(run_dir, "probe_lens_context_shard*.json"))):
        for c in json.load(open(p)):
            if int(c["sample"]) == sample:
                return c
    return None


def parse_steps(spec, d_conv):
    """'1,3,5,-1' (1-based forwards; -1 = last) plus 'R' for the reading forward -> 0-based d values."""
    out = []
    for tok in spec.split(","):
        t = tok.strip()
        if not t:
            continue
        if t.upper() == "R":
            out.append(D_READING)
        elif t == "-1":
            out.append(d_conv)
        else:
            out.append(int(t) - 1)
    seen, keep = set(), []
    for d in out:
        if d not in seen and (d == D_READING or 0 <= d <= d_conv):
            keep.append(d)
            seen.add(d)
    return keep


def parse_range(spec, lo_default, hi_default):
    if not spec or spec == "all":
        return lo_default, hi_default
    a, _, b = spec.partition(":")
    return (int(a) if a else lo_default), (int(b) if b else hi_default)


def tok_str(tokenizer, tid, mask_id, maxlen=10):
    if tid is None or int(tid) < 0:
        return "?"
    tid = int(tid)
    if tid == mask_id:
        return "[M]"
    if tokenizer is None:
        return str(tid)
    s = tokenizer.decode([tid])
    s = s.replace("\n", "\\n").replace("\t", "\\t")
    s = s.strip() or ("_" if s else "''")     # "_" = a bare space token
    return s[:maxlen]


def load_tokenizer(path):
    """Load a tokenizer from `path`; if that fails, try the un-merged sibling checkpoint (the moe
    convertor does not always copy the tokenizer files into the merged dir)."""
    if not path:
        return None
    from transformers import AutoTokenizer
    cands = [path]
    if path.rstrip("/").endswith("-moe-merge") or path.rstrip("/").endswith("-merged"):
        cands.append(path.rstrip("/").rsplit("-moe-merge", 1)[0].rsplit("-merged", 1)[0])
    for c in cands:
        try:
            return AutoTokenizer.from_pretrained(c, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            print(f"[dash] tokenizer not loadable from {c}: {type(e).__name__}")
    print("[dash] WARNING: no tokenizer -- cells will show raw token ids. Point --tokenizer_path at a "
          "directory containing tokenizer.json / tokenizer_config.json (the ORIGINAL download works).")
    return None


def draw_strip(ax, labels, n_pos, title, face, edge, fontsize, highlight=None, xticklabels=None):
    """A one-row label strip (converged tokens on top / fed tokens at the bottom)."""
    ax.set_xlim(-0.5, n_pos - 0.5)
    ax.set_ylim(-0.5, 0.5)
    if xticklabels is None:
        ax.set_xticks([])
    else:
        ax.set_xticks(range(n_pos))
        ax.set_xticklabels(xticklabels, fontsize=max(fontsize - 1.0, 5.0))
        ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for i, lab in enumerate(labels):
        fc = face if (highlight is None or not highlight[i]) else "#ffe0b2"
        ax.add_patch(plt.Rectangle((i - 0.5, -0.5), 1, 1, facecolor=fc, edgecolor=edge, lw=0.4))
        ax.text(i, 0, lab, ha="center", va="center", fontsize=fontsize, color="#212121")
    ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=fontsize + 1, labelpad=8)


def main():
    ap = argparse.ArgumentParser(description="Per-block lens dashboard (one sample, one block)")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--sample", type=int, required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--steps", default="1,3,5,-1", help="1-based forwards; -1 = last; R = reading forward")
    ap.add_argument("--layers", default="1:", help="layer slice, e.g. 1: (skip embeddings) or 10:21")
    ap.add_argument("--pos_range", default="all", help="block position slice, e.g. 0:16")
    ap.add_argument("--metric", default="nlp_tc", choices=list(METRICS))
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--mask_id", type=int, default=MASK_ID_DEFAULT)
    ap.add_argument("--cell_w", type=float, default=None, help="inches; default: fit to the token text")
    ap.add_argument("--cell_h", type=float, default=None, help="inches; default: fit topk lines of text")
    ap.add_argument("--fontsize", type=float, default=12.5, help="cell font size")
    ap.add_argument("--header_fontsize", type=float, default=16.0, help="prompt/header font size")
    ap.add_argument("--font", default=None, help="force a font family (must cover CJK for this vocab)")
    ap.add_argument("--show_probs", action="store_true", help="append each token's probability")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    setup_font(args.font)
    D = load_block(args.run_dir, args.sample, args.block)
    ctx = load_context(args.run_dir, args.sample)
    tokenizer = load_tokenizer(args.tokenizer_path)

    d_conv = int(D["d_conv"][0])
    steps = parse_steps(args.steps, d_conv)
    if not steps:
        raise SystemExit(f"no valid steps (block has forwards 1..{d_conv + 1})")
    n_asked = len([t for t in args.steps.split(",") if t.strip()])
    if len(steps) < n_asked:
        print(f"[dash] note: this block has only {d_conv + 1} decode forwards -- "
              f"{n_asked - len(steps)} requested step(s) dropped")
    lo_l, hi_l = parse_range(args.layers, int(D["layer"].min()), int(D["layer"].max()) + 1)
    layers = np.arange(max(lo_l, int(D["layer"].min())), min(hi_l, int(D["layer"].max()) + 1))
    all_pos = np.unique(D["pos"])
    lo_p, hi_p = parse_range(args.pos_range, int(all_pos.min()), int(all_pos.max()) + 1)
    positions = all_pos[(all_pos >= lo_p) & (all_pos < hi_p)]
    nL, nP = len(layers), len(positions)
    lidx = {l: i for i, l in enumerate(layers)}
    pidx = {p: i for i, p in enumerate(positions)}

    label, transform, vmax_fixed = METRICS[args.metric]
    val_all = np.asarray(transform(D), dtype=np.float32)

    # per-step grids
    panels = []
    for d in steps:
        m = D["d"] == d
        vals = np.full((nL, nP), np.nan, dtype=np.float32)
        top = np.empty((nL, nP), dtype=object)
        rows = np.nonzero(m)[0]
        for r in rows:
            l, p = int(D["layer"][r]), int(D["pos"][r])
            if l not in lidx or p not in pidx:
                continue
            i, j = lidx[l], pidx[p]
            vals[i, j] = val_all[r]
            ids, probs = D["top_ids"][r][:args.topk], D["top_probs"][r][:args.topk]
            tc = int(D["t_c"][r])
            lines = []
            for tid, pr in zip(ids, probs):
                s = tok_str(tokenizer, tid, args.mask_id)
                if int(tid) == tc:
                    s = f"[{s}]"
                lines.append(f"{s} {pr:.2f}" if args.show_probs else s)
            top[i, j] = "\n".join(lines)
        # strips
        tc_lab, in_lab, in_masked = [], [], []
        for p in positions:
            rr = np.nonzero(m & (D["pos"] == p))[0]
            if rr.size == 0:
                tc_lab.append("")
                in_lab.append("")
                in_masked.append(False)
                continue
            r = rr[0]
            tc_lab.append(tok_str(tokenizer, int(D["t_c"][r]), args.mask_id))
            if D["_have_in_tok"] and int(D["in_tok"][r]) >= 0:
                it = int(D["in_tok"][r])
            else:   # older shards: infer from the state flag
                it = args.mask_id if int(D["state"][r]) == 0 else int(D["t_c"][r])
            in_lab.append(tok_str(tokenizer, it, args.mask_id))
            in_masked.append(it == args.mask_id)
        name = "reading fwd (clean t_c in)" if d == D_READING else f"forward {d + 1}"
        if d == d_conv:
            name += " = last"
        panels.append(dict(d=d, name=name, vals=vals, top=top,
                           tc=tc_lab, inp=in_lab, in_masked=np.array(in_masked)))

    finite = np.concatenate([p["vals"][np.isfinite(p["vals"])] for p in panels])
    vmax = vmax_fixed if vmax_fixed is not None else float(np.nanpercentile(finite, 98)) or 1.0
    cmap = matplotlib.colormaps["Blues"].copy()
    cmap.set_bad("#f5f5f5")
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=max(vmax, 1e-6))

    # ---- figure ----
    # ---- geometry driven by the font: cells are sized to fit their text ----
    fs_cell = float(args.fontsize)
    fs_strip = fs_cell + 1.0
    n_lines_cell = max(args.topk, 1)
    longest = 1
    for pan in panels:
        for cell in pan["top"].ravel():
            if cell:
                longest = max(longest, max(len(ln) for ln in cell.split("\n")))
    longest = min(longest, 14)
    cell_h = args.cell_h or (n_lines_cell * fs_cell * 1.32 / 72.0 + 0.10)
    cell_w = args.cell_w or (max(longest + 1.5, 6.0) * fs_cell * 0.60 / 72.0)

    # ---- header text (built after the width is known, so it wraps to the figure) ----
    head = (f"sample {args.sample}, block {args.block}  |  {d_conv + 1} decode forwards  |  "
            f"metric: {label}")
    if ctx is not None:
        bl = ctx["block_length"]
        bstart = ctx["first_block_start"] + args.block * bl
        head = (f"sample {args.sample}, block {args.block}  |  block positions "
                f"{bstart}..{bstart + bl - 1}  |  {d_conv + 1} decode forwards  |  metric: {label}")
        if tokenizer is not None:
            prompt = tokenizer.decode(ctx["prompt_ids"], skip_special_tokens=False)
            gen_before = ctx["gen_ids"][:max(0, bstart - ctx["P"])]
            gtxt = tokenizer.decode(gen_before, skip_special_tokens=False) if gen_before else "(none)"
            body = f"PROMPT\n{prompt}\n\nGENERATED BEFORE THIS BLOCK\n{gtxt}"
        else:
            body = (f"(context found: prompt is {len(ctx['prompt_ids'])} tokens -- pass --tokenizer_path "
                    f"to render it as text)")
    else:
        body = ("(context json not found in this run -- re-run the probe to record the prompt; "
                "the per-block grids below are unaffected)")
    body = body[:2000]
    panel_w = nP * cell_w
    fig_w = len(panels) * panel_w + 2.6
    fs_head_body = float(args.header_fontsize)
    fs_head_title = fs_head_body + 3.0
    wrap_cols = max(40, int((fig_w - 0.8) * 72.0 / (fs_head_body * 0.60)))
    n_lines = sum(max(1, int(np.ceil(len(ln) / wrap_cols))) for ln in body.split("\n"))
    line_in = fs_head_body * 1.45 / 72.0
    head_h = min(9.0, fs_head_title * 2.2 / 72.0 + line_in * (n_lines + 1))

    grid_h = (nL + 2) * cell_h
    fig_h = head_h + grid_h + 0.9
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = gridspec.GridSpec(2, 1, height_ratios=[head_h, grid_h], hspace=0.05,
                              left=0.055, right=0.965, top=0.975, bottom=0.03)

    axp = fig.add_subplot(outer[0])
    axp.axis("off")
    wrapped = "\n".join("\n".join(ln[i:i + wrap_cols] for i in range(0, max(len(ln), 1), wrap_cols))
                        for ln in body.split("\n"))
    title_frac = (fs_head_title * 2.0 / 72.0) / max(head_h, 1e-6)
    axp.text(0, 1.0, head, fontsize=fs_head_title, fontweight="bold", va="top", family="monospace")
    axp.text(0, max(0.0, 1.0 - title_frac), wrapped, fontsize=fs_head_body, va="top",
             family="monospace", color="#333333", linespacing=1.45)

    inner = gridspec.GridSpecFromSubplotSpec(1, len(panels), subplot_spec=outer[1], wspace=0.035)
    for k, pan in enumerate(panels):
        sub = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=inner[k],
                                               height_ratios=[1, nL, 1], hspace=0.035)
        ax_t = fig.add_subplot(sub[0])
        ax_m = fig.add_subplot(sub[1])
        ax_b = fig.add_subplot(sub[2])

        draw_strip(ax_t, pan["tc"], nP, "converged\nt_c" if k == 0 else "", "#e8f5e9", "#a5d6a7", fs_strip)
        ax_t.set_title(pan["name"], fontsize=fs_strip + 2.0, fontweight="bold", pad=6)

        ax_m.imshow(pan["vals"], aspect="auto", origin="lower", cmap=cmap, norm=norm,
                    extent=[-0.5, nP - 0.5, -0.5, nL - 0.5])
        ax_m.set_xticks([])                      # position numbers live under the input strip
        if k == 0:
            ax_m.set_yticks(range(nL))
            ax_m.set_yticklabels([str(l) for l in layers], fontsize=fs_strip)
            ax_m.set_ylabel("layer", fontsize=fs_strip + 1.0)
        else:
            ax_m.set_yticks([])
        for i in range(nL):
            for j in range(nP):
                if pan["top"][i, j] is None:
                    continue
                v = pan["vals"][i, j]
                dark = np.isfinite(v) and norm(v) > 0.55
                ax_m.text(j, i, pan["top"][i, j], ha="center", va="center",
                          fontsize=fs_cell, linespacing=1.28,
                          color="#ffffff" if dark else "#1a1a1a")
        for j in range(nP + 1):
            ax_m.axvline(j - 0.5, color="#ffffff", lw=0.6)
        for i in range(nL + 1):
            ax_m.axhline(i - 0.5, color="#ffffff", lw=0.6)

        draw_strip(ax_b, pan["inp"], nP, "fed input" if k == 0 else "", "#e3f2fd", "#90caf9",
                   fs_strip, highlight=pan["in_masked"],
                   xticklabels=[str(p) for p in positions])
        ax_b.set_xlabel("block position", fontsize=fs_strip)

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.972, 0.06, 0.008, 0.5])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label, fontsize=fs_strip)
    cb.ax.tick_params(labelsize=fs_strip - 1.0)

    out = args.out or os.path.join(args.run_dir, "analysis",
                                   f"dashboard_s{args.sample}_b{args.block}_{args.metric}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[dash] wrote {out}  ({fig_w:.0f}x{fig_h:.0f} in @ {args.dpi} dpi; "
          f"{nL} layers x {nP} positions x {len(panels)} steps)")


if __name__ == "__main__":
    main()
