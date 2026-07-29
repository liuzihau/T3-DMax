# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# TAP-vs-BASE recall eval (the probe, extended with the trained DARC head). For each block, ONE forward with
# the block masked and prior blocks COMMITTED (= the model's own 0.5-decode == self-gold, so "reveal-prior with
# gold prior" is identical to the faithful decode's forward-1). Records, per in-block position, recall@K (top-k)
# and recall@l% (percentile) against the block's converged gold, for SIX curves:
#     base {L18,L19,L20} = base logit-lens hs[18]/hs[19]/out.logits
#     tap  {L18,L19,L20} = DARC head readout / fuse->replay-1-layer / ->final
# Focus is layers >= 18. Output: runs/tap_recall.npz (+ _meta.json). Plot with `--plot`.
#
# Run:
#   cd dInfer/evaluations
#   python eval_tap_recall.py --model_path ../../DMax-16B-merge \
#       --head_dir ../../dFactory/darc_runs/trial_arout --limit 150 --out runs/tap_recall.npz
#   python eval_tap_recall.py --plot --out runs/tap_recall.npz     # -> runs/plots/tap_vs_base_recall.png

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_T3 = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_HERE, os.path.join(_T3, "dFactory", "models", "darc"), os.path.join(_T3, "dFactory", "tasks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

K_ABS = [1, 5, 10, 50, 100, 500]
K_PCT = [0.0001, 0.001, 0.01, 0.05, 0.10]
CURVES = ["base18", "base19", "base20", "tap18", "tap19", "tap20"]


def _latest_ckpt(head_dir):
    cks = sorted(glob.glob(os.path.join(head_dir, "head_step*.pt")),
                 key=lambda p: int(p.split("head_step")[-1].split(".pt")[0]))
    return cks[-1] if cks else os.path.join(head_dir, "head_final.pt")


@torch.no_grad()
def run_eval(args):
    from transformers import AutoTokenizer
    from probe_layer_readout import load_fused, decode_and_maybe_probe
    from dinfer.decoding.generate_t3d import build_block_causal_mask
    from dinfer.decoding.generate_dbet import MASK_ID
    from configuration_darc import DarcConfig
    from modeling_darc import DarcHead
    from train_darc import replay_levels
    from eval_tasks import load_task

    device = torch.device(args.device)
    mp = os.path.abspath(args.model_path)
    tok = AutoTokenizer.from_pretrained(os.path.abspath(args.tokenizer_path or mp), trust_remote_code=True)
    model = load_fused(mp, device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    embed = model.get_input_embeddings(); lm_head = model.get_output_embeddings()
    final_norm = model.model.norm; rotary_emb = model.model.rotary_emb
    V = int(model.config.vocab_size); dt = embed.weight.dtype

    ck_path = args.head_ckpt or _latest_ckpt(args.head_dir)
    ck = torch.load(ck_path, map_location=device)
    cfg = DarcConfig()
    for k, v in ck["config"].items():
        setattr(cfg, k, v)
    head = DarcHead(cfg).to(device=device, dtype=dt).eval()
    head.load_state_dict(ck["state_dict"], strict=False)
    B = cfg.block_size; tap = cfg.tap_hidden_index
    print(f"[eval] model={os.path.basename(mp)} head={ck_path} tap_index={tap} block={B} "
          f"loss2_inject={getattr(cfg,'loss2_inject','?')}")

    count = np.zeros(B, dtype=np.float64)
    hit_abs = {c: np.zeros((len(K_ABS), B)) for c in CURVES}
    hit_pct = {c: np.zeros((len(K_PCT), B)) for c in CURVES}

    def accum(name, logits, gold, rel):                    # logits [n,V], gold [n], rel [n] block-rel positions
        gl = logits.float().gather(1, gold[:, None])
        rank = (logits.float() > gl).sum(1).cpu().numpy()   # 0-indexed rank of the gold token
        for i, r in enumerate(rel):
            for ki, K in enumerate(K_ABS):
                hit_abs[name][ki, r] += rank[i] < K
            for ki, q in enumerate(K_PCT):
                hit_pct[name][ki, r] += rank[i] < max(1, int(q * V))

    rows = load_task(args.task, limit=args.limit)
    import time
    t0 = time.time()
    for ridx, row in enumerate(rows):
        pid = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}], add_generation_prompt=True,
                                      tokenize=True, return_tensors="pt").to(device)
        P = pid.shape[1]
        gold_x, gold_eos = decode_and_maybe_probe(model, embed, lm_head, final_norm, pid, args.gen_length, B,
                                                  0.5, V, device, gold_block_fn=None, acc=None)
        ge = int(gold_eos)
        for bs in range((P // B) * B, ge, B):              # includes the partial first block
            be = bs + B
            r = max(bs, P); gen_end = min(be, ge)
            if r >= gen_end:
                continue                                    # block has no generated tokens
            n = gen_end - r                                 # scored generated positions [r, gen_end)
            rel = list(range(r - bs, gen_end - bs))         # block-relative positions
            gold_g = gold_x[r:gen_end]

            noisy = gold_x[:be].clone(); noisy[r:] = MASK_ID   # reveal [0,r) committed; mask block gen + future
            pos = torch.arange(be, device=device)[None]
            attn = build_block_causal_mask(be, B, dtype=dt, device=device)
            out = model(input_ids=noisy[None], attention_mask=attn, position_ids=pos,
                        use_cache=False, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states
            # base curves (readout at the scored generated positions)
            accum("base18", lm_head(final_norm(hs[tap][0, r:gen_end])), gold_g, rel)
            accum("base19", lm_head(final_norm(hs[tap + 1][0, r:gen_end])), gold_g, rel)
            accum("base20", out.logits[0, r:gen_end], gold_g, rel)
            # tap curves: head on the block -> ar_out -> inject -> replay
            cos, sin = rotary_emb(hs[tap], pos)
            with torch.autocast(device_type="cuda", dtype=dt):
                _, _, gen_logits, gen_pos, ar_out = head.forward_train(
                    hs[tap][:, bs:be], noisy[bs:be][None], noisy[bs:be][None],
                    cos[:, bs:be], sin[:, bs:be], embed, final_norm, lm_head, return_gen=True)
            accum("tap18", gen_logits[0, :n], gold_g, rel)  # gen_pos contiguous from (r-bs); first n are [r,gen_end)
            h2 = hs[tap].clone()
            for j, p in enumerate(gen_pos):
                h2[0, bs + p] = ar_out[0, j]
            with torch.autocast(device_type="cuda", dtype=dt):
                lv = replay_levels(model, h2, tap, attn, pos, (cos, sin))
            accum("tap19", lm_head(final_norm(lv[tap + 1][0, r:gen_end])), gold_g, rel)
            accum("tap20", lm_head(final_norm(lv[tap + 2][0, r:gen_end])), gold_g, rel)
            for rr in rel:
                count[rr] += 1
        if ridx < 3 or (ridx + 1) % 25 == 0:
            print(f"[{ridx+1}/{len(rows)}] gold_eos={ge} elapsed={time.time()-t0:.0f}s")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out, count=count,
                        **{f"abs_{c}": hit_abs[c] for c in CURVES}, **{f"pct_{c}": hit_pct[c] for c in CURVES})
    meta = dict(model=os.path.basename(mp), head_ckpt=ck_path, tap_index=tap, block_size=B, vocab=V,
                n_examples=len(rows), K_ABS=K_ABS, K_PCT=K_PCT, task=args.task, gen_length=args.gen_length,
                loss2_inject=getattr(cfg, "loss2_inject", "?"))
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[eval] done {time.time()-t0:.0f}s -> {args.out}. Plot: python eval_tap_recall.py --plot --out {args.out}")


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = np.load(args.out); meta = json.load(open(os.path.splitext(args.out)[0] + "_meta.json"))
    cnt = d["count"]; B = meta["block_size"]; Ka = meta["K_ABS"]; Kp = meta["K_PCT"]
    pos = np.arange(B); valid = cnt > 0
    colors = {"18": "tab:blue", "19": "tab:orange", "20": "tab:green"}

    def rec_abs(c, ki):
        return np.where(valid, d[f"abs_{c}"][ki] / np.maximum(cnt, 1), np.nan)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    # (1) recall@1 by in-block position: base (dashed) vs tap (solid), per layer
    for lyr in ("18", "19", "20"):
        ax[0].plot(pos, rec_abs("base" + lyr, 0), "--", color=colors[lyr], marker="o", ms=3, alpha=0.8,
                   label=f"base L{lyr}")
        ax[0].plot(pos, rec_abs("tap" + lyr, 0), "-", color=colors[lyr], marker="s", ms=3,
                   label=f"tap L{lyr}")
    ax[0].set_xlabel("in-block position"); ax[0].set_ylabel("recall@1"); ax[0].set_ylim(0, 1)
    ax[0].set_title(f"recall@1 by position — base (dashed) vs DARC tap (solid)\n{meta['n_examples']} ex, "
                    f"tap=hs[{meta['tap_index']}], inject={meta['loss2_inject']}")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8, ncol=3)
    # (2) recall@K vs K (log) at a few positions, base vs tap, L18 & L20
    sel = [p for p in (1, 4, 8, 16) if p < B and valid[p]]
    for p in sel:
        ax[1].plot(Ka, [d["abs_base20"][ki, p] / max(cnt[p], 1) for ki in range(len(Ka))], "--",
                   marker="o", ms=3, alpha=0.7, label=f"base L20 p{p}")
        ax[1].plot(Ka, [d["abs_tap20"][ki, p] / max(cnt[p], 1) for ki in range(len(Ka))], "-",
                   marker="s", ms=3, label=f"tap L20 p{p}")
    ax[1].set_xscale("log"); ax[1].set_xticks(Ka); ax[1].set_xticklabels(Ka)
    ax[1].set_xlabel("candidate-set size K (top-K)"); ax[1].set_ylabel("recall@K"); ax[1].set_ylim(0, 1.02)
    ax[1].set_title("recall@K vs K at L20 (final): base vs tap"); ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=7, ncol=2)
    out = os.path.join(os.path.dirname(os.path.abspath(args.out)), "plots", "tap_vs_base_recall.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130); print(f"[plot] -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default=None); p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--head_dir", default=None, help="dir with head_step*.pt (uses the latest)")
    p.add_argument("--head_ckpt", default=None, help="explicit head checkpoint (overrides --head_dir)")
    p.add_argument("--task", default="gsm8k"); p.add_argument("--limit", type=int, default=150)
    p.add_argument("--gen_length", type=int, default=256)
    p.add_argument("--out", default="runs/tap_recall.npz"); p.add_argument("--device", default="cuda")
    p.add_argument("--plot", action="store_true", help="read --out npz and write the plot (no model needed)")
    args = p.parse_args()
    if args.plot:
        plot(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
