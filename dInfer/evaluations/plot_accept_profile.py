# Copyright 2026 University of Sydney. Apache-2.0.
# Position-wise drafter acceptance profile (DSpark-style) from eval_dbet_gsm8k --accept_diag records.
#
# Each jsonl record = one drafter EXTEND chain: {"reveal": tokens revealed in the block BEFORE the chain,
# "slots", "conf", "draft_top3" [len,3], "heavy_next" (dummy-heavy argmax = one-step acceptance target),
# "golden" (block tokens at convergence, on-policy)}.
#
# Figures (PNG + TSV):
#   (a) acc @ chain position k vs the ONE-STEP heavy target — unconditional and PREFIX-CONDITIONED
#       (only chains whose positions <k were all top-1 correct; the speculative-acceptance semantics),
#       top-1/2/3;
#   (b) the same vs the CONVERGED block (eventual correctness) — the gap to (a) = the self-repair /
#       confidence-ramp factor per position;
#   (c) prefix-conditioned top-1 one-step acc @ k, split by REVEAL-LEVEL quartile of the block;
#   (d) accepted-prefix-length histogram (first top-1 mismatch vs heavy_next) + mean.
#
#   python evaluations/plot_accept_profile.py --diag accept.jsonl [--out accept_profile] [--kmax 10]

import argparse
import json
import math


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def ok_arrays(ev, topk=1, ref="heavy_next"):
    """Per-position correctness of the chain vs `ref`: draft top-`topk` contains the ref token."""
    return [ref_tok in ev["draft_top3"][k][:topk] for k, ref_tok in enumerate(ev[ref])]


def acc_at_k(events, kmax, topk, ref, conditioned):
    """(acc list, n list) at chain positions 0..kmax-1. `conditioned`: only chains with an all-top1-correct
    prefix vs the SAME ref contribute at position k."""
    num = [0] * kmax
    den = [0] * kmax
    for ev in events:
        ok_t = ok_arrays(ev, topk, ref)
        ok_1 = ok_arrays(ev, 1, ref)
        for k in range(min(len(ok_t), kmax)):
            if conditioned and not all(ok_1[:k]):
                break
            den[k] += 1
            num[k] += int(ok_t[k])
    return [n / d if d else float("nan") for n, d in zip(num, den)], den


def accepted_lengths(events):
    out = []
    for ev in events:
        ok = ok_arrays(ev, 1, "heavy_next")
        first_bad = next((k for k, v in enumerate(ok) if not v), len(ok))
        out.append(first_bad)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diag", required=True)
    p.add_argument("--out", default=None, help="output prefix (default: <diag> without extension)")
    p.add_argument("--kmax", type=int, default=10)
    p.add_argument("--block", type=int, default=32)
    args = p.parse_args()
    out = args.out or args.diag.rsplit(".", 1)[0]

    events = load(args.diag)
    print(f"{len(events)} chains from {len(set(e['ex'] for e in events))} examples; "
          f"mean chain len {sum(len(e['slots']) for e in events)/max(len(events),1):.2f}")

    K = args.kmax
    rows = []
    curves = {}
    for ref, tag in (("heavy_next", "1step"), ("golden", "golden")):
        for cond in (False, True):
            for topk in (1, 2, 3):
                acc, den = acc_at_k(events, K, topk, ref, cond)
                curves[(tag, cond, topk)] = (acc, den)
                rows.append([f"{tag}_{'cond' if cond else 'uncond'}_top{topk}"] +
                            [f"{a:.4f}" if a == a else "nan" for a in acc])
    lens = accepted_lengths(events)
    mean_len = sum(lens) / max(len(lens), 1)
    print(f"mean accepted prefix length (vs one-step heavy): {mean_len:.2f}")

    # reveal quartiles (fraction of the block revealed when the chain launched)
    qs = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    reveal_curves = {}
    for lo, hi in qs:
        sub = [e for e in events if lo <= e["reveal"] / args.block < hi]
        acc, den = acc_at_k(sub, K, 1, "heavy_next", True)
        reveal_curves[(lo, hi)] = (acc, den, len(sub))

    with open(out + ".tsv", "w") as fh:
        fh.write("\t".join(["curve"] + [f"k{k}" for k in range(K)]) + "\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
        fh.write("\t".join(["n_cond_1step"] + [str(d) for d in curves[("1step", True, 1)][1]]) + "\n")
        fh.write(f"mean_accepted_len\t{mean_len:.3f}\n")
    print(f"[tsv] -> {out}.tsv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable -- TSV only")
        return

    ks = list(range(K))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, tag, title in ((axes[0][0], "1step", "vs one-step heavy (acceptance)"),
                           (axes[0][1], "golden", "vs converged block (eventual)")):
        for topk, style in ((1, "-o"), (2, "--s"), (3, ":^")):
            ax.plot(ks, curves[(tag, True, topk)][0], style, label=f"top{topk} cond", ms=4)
        ax.plot(ks, curves[(tag, False, 1)][0], "-x", color="gray", label="top1 uncond", ms=4)
        if tag == "1step":
            for k, d in enumerate(curves[(tag, True, 1)][1]):
                if d:
                    ax.annotate(str(d), (k, 0.02), fontsize=6, ha="center")
        ax.set_title(title); ax.set_xlabel("chain position k"); ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axes[1][0]
    for (lo, hi), (acc, den, n) in reveal_curves.items():
        ax.plot(ks, acc, "-o", ms=3, label=f"reveal {int(lo*100)}-{int(hi*100)}% (n={n})")
    ax.set_title("top1 cond one-step acc by reveal level"); ax.set_xlabel("chain position k")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax = axes[1][1]
    mx = max(lens) if lens else 1
    ax.hist(lens, bins=range(0, mx + 2), align="left", rwidth=0.85)
    ax.axvline(mean_len, color="r", ls="--", label=f"mean {mean_len:.2f}")
    ax.set_title("accepted prefix length (first top1 mismatch)"); ax.set_xlabel("tokens")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"{args.diag}  ({len(events)} chains)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=150)
    print(f"[png] -> {out}.png")


if __name__ == "__main__":
    main()
