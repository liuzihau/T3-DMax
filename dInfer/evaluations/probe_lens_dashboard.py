# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# PER-BLOCK LENS DASHBOARD -- the zoom-in view of one (sample, block) from the lens-surface shards.
#
# Layout
#   row 1 : the prompt + the generated prefix into this block, REPEATED above every panel so it stays
#           in view whichever panel you are reading
#   row 2 : one panel per selected denoise step (default 1, 3, 5, last). Each panel reads left to right:
#             left strip  = the token actually FED at that position entering that step ([M] if masked)
#             main grid   = block position (y, 0 at the top) x layer (x, first -> last); cell colour =
#                           divergence to t_c, cell text = that layer's top-k readout tokens (the
#                           converged token is [bracketed] when it appears)
#             right strip = the CONVERGED token t_c for that position
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


FONT_URLS = [   # tried in order by --install_font (no sudo needed; lands in ~/.fonts)
    "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
    "https://cdn.jsdelivr.net/gh/googlefonts/noto-cjk@main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
    "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf",
]
_CJK_FILE_HINTS = ("cjk", "notosanssc", "notosansjp", "notosanstc", "sourcehan", "wqy", "zenhei",
                   "microhei", "droidsansfallback", "msyh", "simhei", "pingfang", "unifont")


def _scan_font_files():
    """Font files on disk, including dirs matplotlib may not have indexed yet."""
    dirs = [os.path.expanduser("~/.fonts"), os.path.expanduser("~/.local/share/fonts"),
            "/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/Library/Fonts")]
    out = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith((".otf", ".ttf")):     # .ttc needs an index -- skip
                    out.append(os.path.join(root, f))
    return out


def install_cjk_font():
    """Download a Noto CJK font into ~/.fonts (no sudo, no package manager). Returns the path or None."""
    import urllib.request
    dest_dir = os.path.expanduser("~/.fonts")
    os.makedirs(dest_dir, exist_ok=True)
    for url in FONT_URLS:
        name = os.path.basename(url).replace("%5B", "[").replace("%5D", "]")
        dest = os.path.join(dest_dir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
            print(f"[dash] font already present: {dest}")
            return dest
        try:
            print(f"[dash] downloading {url} ...")
            urllib.request.urlretrieve(url, dest)
            if os.path.getsize(dest) > 1_000_000:
                print(f"[dash] installed {dest}")
                return dest
            os.remove(dest)
        except Exception as e:  # noqa: BLE001
            print(f"[dash]   failed ({type(e).__name__}: {e})")
    return None


def setup_font(user_font=None, install=False):
    """Make CJK glyphs renderable. Strategy, most reliable first:
      1. --font NAME, if given;
      2. a CJK font file found on disk -> registered in-process with addfont() (no font-cache dance)
         and made the PRIMARY family (Noto CJK covers Latin too, so nothing else is lost);
      3. a CJK family matplotlib already knows about;
      4. optional download into ~/.fonts (--install_font).
    Making it primary (not just a fallback) matters: per-glyph fallback needs matplotlib >= 3.6."""
    from matplotlib import font_manager
    print(f"[dash] matplotlib {matplotlib.__version__}")
    if install:
        install_cjk_font()

    # register any on-disk CJK files matplotlib may not have indexed
    for path in _scan_font_files():
        if any(h in os.path.basename(path).lower() for h in _CJK_FILE_HINTS):
            try:
                font_manager.fontManager.addfont(path)
            except Exception:  # noqa: BLE001
                pass

    avail = {f.name for f in font_manager.fontManager.ttflist}
    picks = [p for p in ([user_font] if user_font else []) + [f for f in CJK_FONTS if f in avail] if p]
    if not picks:   # last resort: any registered family whose file name looked CJK
        for f in font_manager.fontManager.ttflist:
            if any(h in os.path.basename(getattr(f, "fname", "")).lower() for h in _CJK_FILE_HINTS):
                picks.append(f.name)
                break
    if picks:
        for key in ("font.sans-serif", "font.monospace"):
            rest = [x for x in matplotlib.rcParams[key] if x not in picks]
            matplotlib.rcParams[key] = picks[:2] + rest        # PRIMARY, not just fallback
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["axes.unicode_minus"] = False
        print(f"[dash] CJK font in use: {picks[0]}")
        return True
    print("[dash] WARNING: no CJK-capable font found -- Chinese tokens will render as boxes.\n"
          "        easiest fix (no sudo, downloads Noto Sans CJK into ~/.fonts):\n"
          "          python probe_lens_dashboard.py --install_font ...   (same args as usual)\n"
          "        alternatives:\n"
          "          pip install mplfonts && mplfonts init\n"
          "          sudo apt-get install -y fonts-noto-cjk\n"
          "        then:  rm -rf ~/.cache/matplotlib\n"
          "        to see what matplotlib can find:  --list_fonts")
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


def _disable_mathtext():
    """matplotlib treats a string with an EVEN number of '$' as inline mathtext, so GSM8K's dollar
    amounts ("$18 ... $2") get sent to the TeX parser and raise ParseException at draw time.
    rcParams['text.parse_math'] (matplotlib >= 3.5) turns that off globally -- the clean fix.
    Returns True if it worked; otherwise callers fall back to escaping."""
    try:
        matplotlib.rcParams["text.parse_math"] = False
        return True
    except KeyError:
        return False


_MATH_OFF = _disable_mathtext()


def mpl_escape(s):
    """No-op when math parsing is disabled (otherwise the escapes would show up literally);
    escapes '$' on old matplotlib where the rcParam does not exist."""
    if _MATH_OFF:
        return s
    return s.replace("\\", "\\\\").replace("$", r"\$")


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
    return mpl_escape(s[:maxlen])


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


def draw_vstrip(ax, labels, n_pos, title, face, edge, fontsize, highlight=None, yticklabels=None):
    """A one-COLUMN label strip beside the grid: one cell per block position, top to bottom."""
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(n_pos - 0.5, -0.5)                       # position 0 at the top
    ax.set_xticks([])
    if yticklabels is None:
        ax.set_yticks([])
    else:
        ax.set_yticks(range(n_pos))
        ax.set_yticklabels(yticklabels, fontsize=max(fontsize - 1.0, 5.0))
        ax.tick_params(axis="y", length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    for i, lab in enumerate(labels):
        fc = face if (highlight is None or not highlight[i]) else "#ffe0b2"
        ax.add_patch(plt.Rectangle((-0.5, i - 0.5), 1, 1, facecolor=fc, edgecolor=edge, lw=0.4))
        ax.text(0, i, lab, ha="center", va="center", fontsize=fontsize, color="#212121")
    ax.set_title(title, fontsize=fontsize, pad=5)


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
    ap.add_argument("--header_fontsize", type=float, default=20.0, help="prompt/header font size")
    ap.add_argument("--font", default=None, help="force a font family (must cover CJK for this vocab)")
    ap.add_argument("--install_font", action="store_true",
                    help="download Noto Sans CJK into ~/.fonts if no CJK font is available (no sudo)")
    ap.add_argument("--list_fonts", action="store_true", help="print the font families matplotlib sees, exit")
    ap.add_argument("--show_probs", action="store_true", help="append each token's probability")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    setup_font(args.font, install=args.install_font)
    if args.list_fonts:
        from matplotlib import font_manager
        names = sorted({f.name for f in font_manager.fontManager.ttflist})
        print(f"[dash] {len(names)} families visible to matplotlib:")
        for n in names:
            print("   ", n)
        return
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
    # TRANSPOSED layout: y = block position (0 at the top), x = layer (first -> last),
    # with the fed input as the left-most column and the converged token as the right-most.
    panel_w = (nL + 2) * cell_w                 # +2 for the input / converged strips
    fig_w = len(panels) * panel_w + 1.9
    fs_head_body = float(args.header_fontsize)
    fs_head_title = fs_head_body + 3.0
    # the prompt is repeated once per panel, so it wraps to ONE panel's width
    wrap_cols = max(28, int((panel_w - 0.35) * 72.0 / (fs_head_body * 0.60)))
    # wrap FIRST, escape after (escaping before could split a "\$" pair across two lines)
    wrapped = "\n".join(mpl_escape(ln[i:i + wrap_cols])
                        for ln in body.split("\n")
                        for i in range(0, max(len(ln), 1), wrap_cols))
    n_lines = wrapped.count("\n") + 1
    line_in = fs_head_body * 1.45 / 72.0
    head_h = min(11.0, fs_head_title * 2.4 / 72.0 + line_in * (n_lines + 1))

    grid_h = nP * cell_h
    fig_h = head_h + grid_h + 1.5
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = gridspec.GridSpec(2, len(panels), height_ratios=[head_h, grid_h], hspace=0.03,
                              wspace=0.05, left=0.035, right=0.955, top=0.985, bottom=0.035)

    # one figure-wide title (per-panel titles would overlap each other), auto-shrunk to fit the width
    fs_title = min(fs_head_title, max(9.0, (fig_w - 0.8) * 72.0 / (len(head) * 0.60)))
    fig.suptitle(mpl_escape(head), fontsize=fs_title, fontweight="bold", family="monospace", y=0.995)

    for k, pan in enumerate(panels):
        # --- prompt, repeated above every panel so it is always in view ---
        axp = fig.add_subplot(outer[0, k])
        axp.axis("off")
        axp.text(0, 0.88, wrapped, fontsize=fs_head_body, va="top",
                 family="monospace", color="#333333", linespacing=1.45)

        # --- panel: [fed input | layer grid | converged t_c] ---
        sub = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1, k],
                                               width_ratios=[1, nL, 1], wspace=0.02)
        ax_in = fig.add_subplot(sub[0])
        ax_m = fig.add_subplot(sub[1])
        ax_tc = fig.add_subplot(sub[2])

        draw_vstrip(ax_in, pan["inp"], nP, "fed\ninput", "#e3f2fd", "#90caf9", fs_strip,
                    highlight=pan["in_masked"], yticklabels=[str(p) for p in positions])
        if k == 0:
            ax_in.set_ylabel("block position", fontsize=fs_strip + 1.0)

        ax_m.imshow(pan["vals"].T, aspect="auto", origin="upper", cmap=cmap, norm=norm,
                    extent=[-0.5, nL - 0.5, nP - 0.5, -0.5])
        ax_m.set_title(pan["name"], fontsize=fs_strip + 3.0, fontweight="bold", pad=8)
        ax_m.set_yticks([])
        ax_m.set_xticks(range(nL))
        ax_m.set_xticklabels([str(l) for l in layers], fontsize=fs_strip)
        ax_m.tick_params(axis="x", length=0, pad=3)
        ax_m.set_xlabel("layer  (first -> last)", fontsize=fs_strip + 1.0)
        for i in range(nL):
            for j in range(nP):
                if pan["top"][i, j] is None:
                    continue
                v = pan["vals"][i, j]
                dark = np.isfinite(v) and norm(v) > 0.55
                ax_m.text(i, j, pan["top"][i, j], ha="center", va="center",
                          fontsize=fs_cell, linespacing=1.28,
                          color="#ffffff" if dark else "#1a1a1a")
        for i in range(nL + 1):
            ax_m.axvline(i - 0.5, color="#ffffff", lw=0.6)
        for j in range(nP + 1):
            ax_m.axhline(j - 0.5, color="#ffffff", lw=0.6)

        draw_vstrip(ax_tc, pan["tc"], nP, "converged\nt_c", "#e8f5e9", "#a5d6a7", fs_strip)

    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.963, 0.06, 0.006, 0.4])
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
