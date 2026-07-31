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

import numpy as np                                             # torch imported lazily in run_eval (so --plot needs no torch)

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


def run_eval(args):
    import torch
    torch.set_grad_enabled(False)                            # eval-only: no graphs (avoid OOM over many examples)
    from transformers import AutoTokenizer
    from probe_layer_readout import load_fused, decode_and_maybe_probe
    from dinfer.decoding.generate_t3d import build_block_causal_mask
    from dinfer.decoding.generate_dbet import MASK_ID
    from configuration_darc import DarcConfig
    from modeling_darc import DarcHead
    from lora import add_lora, set_lora_enabled, load_lora_state
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

    # LoRA (if this checkpoint trained it): rebuild from the saved A/B (r, alpha inferred) and apply to the same
    # decoder layer(s). Kept OFF by default -> the base forward is the honest FROZEN baseline; toggled ON only
    # around the DARC replay so the tap L19/L20 curves reflect the LoRA-adapted layer (== how it was trained).
    lora_mods = []
    lst, llayers = ck.get("lora"), ck.get("lora_layers")
    if lst and llayers:
        r = int(lst[0]["A"].shape[0]); alpha = int(round(float(lst[0]["scale"]) * r))
        lora_mods = add_lora(model, llayers, r=r, alpha=alpha)
        load_lora_state(lora_mods, lst)
        set_lora_enabled(lora_mods, False)
    print(f"[eval] model={os.path.basename(mp)} head={ck_path} tap_index={tap} block={B} "
          f"n_ar_passes={getattr(cfg,'n_ar_passes',1)} dynamic_k={getattr(cfg,'dynamic_k',False)} "
          f"loss2_inject={getattr(cfg,'loss2_inject','?')} lora_layers={llayers if lora_mods else None}")

    import torch.nn.functional as Fnn
    count = np.zeros(B, dtype=np.float64)
    hit_abs = {c: np.zeros((len(K_ABS), B)) for c in CURVES}
    hit_pct = {c: np.zeros((len(K_PCT), B)) for c in CURVES}
    cos_sum = {L: np.zeros(B) for L in ("18", "19", "20")}   # base-vs-tap post-norm hidden cosine, per position

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
            # base POST-NORM hiddens (input to lm_head) at the scored generated positions
            bn18 = final_norm(hs[tap][0, r:gen_end])
            bn19 = final_norm(hs[tap + 1][0, r:gen_end])
            bn20 = hs[tap + 2][0, r:gen_end]                 # hs[20] is already post-norm
            accum("base18", lm_head(bn18), gold_g, rel)
            accum("base19", lm_head(bn19), gold_g, rel)
            accum("base20", out.logits[0, r:gen_end], gold_g, rel)
            # tap: head on the block -> ar_out -> inject -> replay
            cos, sin = rotary_emb(hs[tap], pos)
            with torch.autocast(device_type="cuda", dtype=dt):
                _, _, gen_logits, gen_pos, ar_out = head.forward_train(
                    hs[tap][:, bs:be], noisy[bs:be][None], noisy[bs:be][None],
                    cos[:, bs:be], sin[:, bs:be], embed, final_norm, lm_head, return_gen=True)
            # build what gets injected as the new hs[tap] -- MUST match how the head was trained (loss2_inject)
            if cfg.loss2_inject == "ar_out":
                inject = ar_out                                                 # [1,n_gen,D] the AR output
            else:                                                              # fuse modes: concat [X, orig h_18]
                h_gen = hs[tap][:, [bs + p for p in gen_pos]]
                x = ar_out if cfg.loss2_inject == "ar_fuse" else \
                    head.soft_embed_topk(head.readout(ar_out, final_norm, lm_head), embed.weight)
                with torch.autocast(device_type="cuda", dtype=dt):
                    inject = head.fuse(x, h_gen)
            h2 = hs[tap].clone()
            for j, p in enumerate(gen_pos):
                h2[0, bs + p] = inject[0, j]
            if lora_mods:
                set_lora_enabled(lora_mods, True)                             # replay through the LoRA-adapted layer
            with torch.autocast(device_type="cuda", dtype=dt):
                lv = replay_levels(model, h2, tap, attn, pos, (cos, sin))
            if lora_mods:
                set_lora_enabled(lora_mods, False)
            tn18 = final_norm(ar_out[0, :n])                 # ar_out[:n] aligns with [r,gen_end)
            tn19 = final_norm(lv[tap + 1][0, r:gen_end])
            tn20 = final_norm(lv[tap + 2][0, r:gen_end])
            accum("tap18", lm_head(tn18), gold_g, rel)
            accum("tap19", lm_head(tn19), gold_g, rel)
            accum("tap20", lm_head(tn20), gold_g, rel)
            # cosine similarity: base vs tap POST-NORM hidden (pre-lm_head), per position
            for lname, bvec, tvec in (("18", bn18, tn18), ("19", bn19, tn19), ("20", bn20, tn20)):
                cs = Fnn.cosine_similarity(bvec.float(), tvec.float(), dim=-1).cpu().numpy()
                for i, rr in enumerate(rel):
                    cos_sum[lname][rr] += cs[i]
            for rr in rel:
                count[rr] += 1
        if ridx < 3 or (ridx + 1) % 25 == 0:
            print(f"[{ridx+1}/{len(rows)}] gold_eos={ge} elapsed={time.time()-t0:.0f}s")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out, count=count,
                        **{f"abs_{c}": hit_abs[c] for c in CURVES}, **{f"pct_{c}": hit_pct[c] for c in CURVES},
                        **{f"cos_{L}": cos_sum[L] for L in ("18", "19", "20")})
    meta = dict(model=os.path.basename(mp), head_ckpt=ck_path, tap_index=tap, block_size=B, vocab=V,
                n_examples=len(rows), K_ABS=K_ABS, K_PCT=K_PCT, task=args.task, gen_length=args.gen_length,
                loss2_inject=getattr(cfg, "loss2_inject", "?"), n_ar_passes=getattr(cfg, "n_ar_passes", 1),
                dynamic_k=getattr(cfg, "dynamic_k", False), lora_layers=(llayers if lora_mods else None))
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[eval] done {time.time()-t0:.0f}s -> {args.out}. Plot: python eval_tap_recall.py --plot --out {args.out}")


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    d = np.load(args.out); meta = json.load(open(os.path.splitext(args.out)[0] + "_meta.json"))
    cnt = d["count"]; Ka = meta["K_ABS"]
    stem = os.path.splitext(os.path.basename(args.out))[0]      # per-npz plot names -> runs never clobber each other
    pdir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "plots")
    POS = [1, 2, 4, 8, 12, 16, 24, 32]                       # 1-indexed in-block positions (2 rows x 4 cols)
    colors = {"18": "tab:blue", "19": "tab:orange", "20": "tab:green"}
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for idx, pp in enumerate(POS):
        ax = axes[idx // 4][idx % 4]
        pr = pp - 1                                          # block-relative index
        if pr >= len(cnt) or cnt[pr] == 0:
            ax.set_title(f"pos {pp} (no data)"); ax.axis("off"); continue
        allv = []
        for lyr in ("18", "19", "20"):                       # 6 lines: L{18,19,20} base (dashed) + tap (solid)
            base = [d[f"abs_base{lyr}"][ki, pr] / cnt[pr] for ki in range(len(Ka))]
            tapv = [d[f"abs_tap{lyr}"][ki, pr] / cnt[pr] for ki in range(len(Ka))]
            ax.plot(Ka, base, "--o", ms=3, color=colors[lyr], alpha=0.75, label=f"L{lyr}")
            ax.plot(Ka, tapv, "-s", ms=3, color=colors[lyr], label=f"L{lyr}+tap")
            allv += base + tapv
        ax.set_xscale("log"); ax.set_xticks(Ka); ax.set_xticklabels(Ka, fontsize=7)
        mn = min([v for v in allv if v > 0] or [0.01])
        if mn >= 0.6:                                        # crowded in 0.6-1.0 -> log-y, cut at 0.9*lowest
            ax.set_yscale("log"); ax.set_ylim(0.9 * mn, 1.02); ax.minorticks_off()
            ax.set_yticks([round(v, 3) for v in np.linspace(0.9 * mn, 1.0, 5)])
            ax.yaxis.set_major_formatter(ScalarFormatter())
        else:
            ax.set_ylim(0, 1.02)
        ax.set_title(f"pos {pp}", fontsize=11); ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("K (top-K)", fontsize=8); ax.set_ylabel("recall@K", fontsize=8)
        if idx == 0:
            ax.legend(fontsize=7, ncol=3, loc="lower right")
    fig.suptitle(f"recall@K vs K by in-block position — base (dashed) vs +tap (solid) at L18/L19/L20   "
                 f"[tap=hs[{meta['tap_index']}], inject={meta.get('loss2_inject')}, "
                 f"passes={meta.get('n_ar_passes', 1)}, lora={meta.get('lora_layers')}, {meta['n_examples']} ex]",
                 fontsize=11)
    out = os.path.join(pdir, f"{stem}_decisionK.png")
    os.makedirs(pdir, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out, dpi=130); print(f"[plot] -> {out}")

    # cosine similarity of base-vs-tap POST-NORM hidden, by position (does the injection survive to L20?)
    if "cos_20" in d:
        fig2, ax2 = plt.subplots(figsize=(9, 5.5))
        vpos = np.where(cnt > 0)[0]
        vals = []
        for L, c in (("18", "tab:blue"), ("19", "tab:orange"), ("20", "tab:green")):
            cs = d[f"cos_{L}"][vpos] / cnt[vpos]
            vals.append(cs)
            ax2.plot(vpos + 1, cs, "-o", ms=3, color=c, label=f"L{L}")
        mn = float(min(v.min() for v in vals)) if len(vpos) else 0.9
        ax2.set_ylim(min(0.9, mn * 0.98), 1.001)
        ax2.set_xlabel("in-block position (1-indexed)"); ax2.set_ylabel("cos(base hidden, tap hidden)")
        ax2.set_title("base-vs-tap POST-NORM hidden cosine, by position\n(→1 = injection washed out; <1 = survives)")
        ax2.grid(alpha=0.3); ax2.legend()
        out2 = os.path.join(pdir, f"{stem}_cossim.png")
        fig2.tight_layout(); fig2.savefig(out2, dpi=130); print(f"[plot] -> {out2}")


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
