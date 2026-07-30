# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# DARC first-trial trainer (SINGLE GPU, plain PyTorch). Trains the ~50M DARC head on a FROZEN DMax-16B using
# the reveal-prior gold data. Reuses the exact components validated by dFactory/scripts/smoke_darc_integration.py
# (load_fused, tap hidden_states[tap_hidden_index], build_block_causal_mask, DarcHead) so training == the smoke.
#
# Efficiency: with reveal_prior only the ACTIVE block needs the head, so after the (batched) frozen heavy
# forward we slice h[:, bs:be] and the head runs 32 positions -- prior context already lives in h.
#
# NOT the multi-GPU VeOmni/FSDP path (train_dbet.py). For a 50M head on a frozen backbone this single-GPU loop
# is enough for the first trial; port to the DBet FSDP harness later if we need scale.
#
# Data: JSONL gold shards from collect_gold_data.py. Train/val split is DETERMINISTIC by rank
# (val = ranks where rank % round(1/val_frac) == 0), so it is stable and extends cleanly as more data is added.
#
# Run:
#   cd dFactory/tasks
#   python train_darc.py --model_path ../../DMax-16B-merge \
#       --gold_glob '../darc_gold/nemotron_math/gold.*.jsonl' --out_dir ../darc_runs/trial1 \
#       --micro_bsz 8 --lr 1e-4 --max_steps 4000 --eval_every 200 --save_every 1000
#   # quick smoke of the loop: add  --max_records 64 --max_steps 20 --eval_every 10

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))                      # dFactory/tasks
_T3 = os.path.abspath(os.path.join(_HERE, "..", ".."))                  # T3-DMax
for _p in (os.path.join(_T3, "dInfer", "evaluations"),
           os.path.join(_T3, "dFactory", "models", "darc"),
           os.path.join(_T3, "dFactory", "tasks", "dataset")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from probe_layer_readout import load_fused                              # noqa: E402
from dinfer.decoding.generate_t3d import build_block_causal_mask        # noqa: E402
from dinfer.decoding.generate_dbet import MASK_ID                       # noqa: E402
from configuration_darc import DarcConfig                              # noqa: E402
from modeling_darc import DarcHead                                     # noqa: E402
from lora import (add_lora, set_lora_enabled, lora_parameters,         # noqa: E402
                  lora_state_dict, load_lora_state)
from data_transform_darc import process_darc_gold_example, iter_gold_records  # noqa: E402


def collate(insts, device):
    st = lambda k: torch.stack([x[k] for x in insts]).to(device, non_blocking=True)
    return {"noisy": st("noisy_input_ids"), "labels": st("labels"),
            "active_bs": torch.tensor([int(x["active_bs"]) for x in insts], device=device)}


def instance_stream(records, tf, shuffle, seed):
    order = list(range(len(records)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for i in order:
        for inst in tf(records[i]):
            yield inst


def batched(stream, n):
    buf = []
    for x in stream:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf


from collections import defaultdict

POS_REPORT = (0, 3, 7, 15, 31)                                    # generated-index -> reported pos 1,4,8,16,32


def _new_acc():
    return defaultdict(lambda: [0.0, 0, 0.0])                     # gen_idx -> [hit, count, loss_sum]


def _bucket(acc, logits_ng, gold_ng):
    """logits_ng [n_gen,V], gold_ng [n_gen]: accumulate hit/count/CE by generated-index (= row index)."""
    v = gold_ng != -100
    if not bool(v.any()):
        return
    ce = F.cross_entropy(logits_ng.float(), gold_ng, reduction="none", ignore_index=-100)   # [n_gen]
    hit = (logits_ng.argmax(-1) == gold_ng).float()
    hitc, cec, vc = hit.cpu(), ce.cpu(), v.cpu()
    for gi in range(gold_ng.shape[0]):
        if bool(vc[gi]):
            a = acc[gi]; a[0] += float(hitc[gi]); a[1] += 1; a[2] += float(cec[gi])


def _report(acc):
    return {gi: (acc[gi][0] / acc[gi][1], acc[gi][2] / acc[gi][1]) for gi in POS_REPORT if acc.get(gi, [0, 0])[1]}


def _fmt_pos(rep, which=0):                                       # which: 0=acc, 1=loss;  pos 1,4,8,16,32
    return " ".join(f"{rep[gi][which]:5.2f}" if gi in rep else "    -" for gi in POS_REPORT)


def replay_levels(model, h, tap_index, attn, pos, cos_sin):
    """Run FROZEN top layers on h; return {level: hidden} for level = tap_index+1 .. num_layers (pre-norm).
    Grad flows through h (into the fuse) if h requires grad; layer weights stay frozen."""
    hs = {}
    for k, layer in enumerate(model.model.layers[tap_index:]):
        h = layer(h, attention_mask=attn, position_ids=pos, position_embeddings=cos_sin, use_cache=False)[0]
        hs[tap_index + k + 1] = h
    return hs


def replay_top(model, h, tap_index, attn, pos, cos_sin, final_norm, lm_head):   # kept for the smoke's check (D)
    hs = replay_levels(model, h, tap_index, attn, pos, cos_sin)
    return lm_head(final_norm(hs[max(hs)]))


def run_micro_batch(model, head, batch, cfg, tap_index, embed, final_norm, lm_head, rotary_emb, dt, device,
                    loss1_w=1.0, loss2_w=0.0, val=False, collect_pos=False, lora_mods=None):
    """Heavy forward -> tap -> per-block head (Loss-1) [+ fuse->replay->final (Loss-2)]. Returns
    (loss, agg, accs). accs maps curve -> position accumulator; curves: train -> tap18(/tap20 if Loss-2);
    val -> the SIX {base18,base19,base20,tap18,tap19,tap20} (base = base logit-lens, no head; tap = with head:
    18 = AR readout=Loss-1, 20 = fuse->replay=Loss-2, 19 = fuse->replay-1-layer). Readout only at the generated
    positions (never [N,L,V]) to stay memory-safe."""
    noisy, labels, act = batch["noisy"], batch["labels"], batch["active_bs"]
    N, L = noisy.shape
    B = cfg.block_size
    do2 = loss2_w > 0                                             # Loss-2 injects the AR block's ar_out (no fuse)
    L18, L19, L20 = tap_index, tap_index + 1, tap_index + 2
    if lora_mods:
        set_lora_enabled(lora_mods, False)                        # base forward = honest FROZEN baseline (no LoRA)
    with torch.no_grad():                                          # frozen backbone
        attn = build_block_causal_mask(L, B, dtype=dt, device=device).expand(N, 1, L, L)
        pos = torch.arange(L, device=device)[None].expand(N, L)
        out = model(input_ids=noisy, attention_mask=attn, position_ids=pos,
                    use_cache=False, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[tap_index]                          # [N,L,D]
        cos, sin = rotary_emb(h, pos)

    want = collect_pos or val
    curves = (["base18", "base19", "base20", "tap18", "tap19", "tap20"] if val
              else (["tap18", "tap20"] if do2 else ["tap18"]))
    accs = {c: _new_acc() for c in curves} if want else {}
    agg = {"loss1": 0.0, "acc1": 0.0, "n_sup": 0, "loss2": 0.0, "acc2": 0.0}
    loss1 = 0.0
    fvals, fni, fpi = [], [], []                                  # grad-safe scatter of fused values into h2
    spans = []                                                    # (n, full_gen_positions[list], gold_ng)
    for n in range(N):
        bs = int(act[n]); be = bs + B
        with torch.autocast(device_type="cuda", dtype=dt):
            l, m, gen_logits, gen_pos, ar_out = head.forward_train(
                h[n:n + 1, bs:be], noisy[n:n + 1, bs:be], labels[n:n + 1, bs:be],
                cos[n:n + 1, bs:be], sin[n:n + 1, bs:be], embed, final_norm, lm_head, return_gen=True)
            if (do2 or val) and gen_pos:
                if cfg.loss2_inject == "ar_out":                  # inject the AR residual output directly
                    vals = ar_out[0]                                        # [n_gen,D] grad -> attn+mlp (g_0=h)
                else:                                             # fuse modes: concat [X, original h_18] -> D
                    h_gen = h[n:n + 1, [bs + p for p in gen_pos]]          # [1,n_gen,D] the untouched tap (h_18)
                    if cfg.loss2_inject == "ar_fuse":             # X = h_ar (hidden-space): keeps the Loss-1
                        x = ar_out                                #   sharp decode AND, w/ h_18, restores hidden info
                    else:                                         # "soft": X = top-k soft-embed (embedding-space)
                        x = head.soft_embed_topk(head.readout(ar_out, final_norm, lm_head), embed.weight)
                    vals = head.fuse(x, h_gen)[0]                          # [n_gen,D] grad -> fuse AND head
                fvals.append(vals); fni += [n] * len(gen_pos); fpi += [bs + p for p in gen_pos]
        loss1 = loss1 + l
        agg["loss1"] += m["loss1"]; agg["acc1"] += m["acc1"]; agg["n_sup"] += m["n_sup"]
        full_pos = [bs + p for p in gen_pos]
        gold_ng = labels[n, full_pos] if full_pos else labels[n, 0:0]
        spans.append((n, full_pos, gold_ng))
        if want and gen_logits is not None:
            _bucket(accs["tap18"], gen_logits[0], gold_ng)        # tap L18 = AR readout (== Loss-1)
    loss1 = loss1 / N
    loss = loss1_w * loss1

    if do2 or val:
        # GRAD-SAFE: functional index_put keeps h2 in the graph so Loss-2's grad reaches the fuse (an in-place
        # write into a no-grad clone would silently drop it -> fuse never trains -> with-L19/L20 == no-tap).
        if fvals:
            h2 = h.detach().index_put(
                (torch.tensor(fni, device=device), torch.tensor(fpi, device=device)),
                torch.cat(fvals, dim=0))
        else:
            h2 = h.detach()
        if lora_mods:
            set_lora_enabled(lora_mods, True)                     # DARC replay: the LoRA-adapted layer consumes h_ar
        with torch.autocast(device_type="cuda", dtype=dt):
            lv = replay_levels(model, h2, tap_index, attn, pos, (cos, sin))   # {L19:h, L20:h}
        if lora_mods:
            set_lora_enabled(lora_mods, False)                    # restore OFF (default)
        loss2 = 0.0
        for (n, full_pos, gold_ng) in spans:
            if not full_pos:
                continue
            with torch.autocast(device_type="cuda", dtype=dt):
                lg20 = lm_head(final_norm(lv[L20][n, full_pos])).float()      # [n_gen,V] grad
            loss2 = loss2 + F.cross_entropy(lg20, gold_ng, ignore_index=-100)
            with torch.no_grad():
                v = gold_ng != -100
                agg["acc2"] += float((lg20.argmax(-1)[v] == gold_ng[v]).float().mean()) if bool(v.any()) else 0.0
                agg["loss2"] += float(F.cross_entropy(lg20, gold_ng, ignore_index=-100).detach())
            if want:
                _bucket(accs["tap20"], lg20.detach(), gold_ng)
            if val:
                _bucket(accs["tap19"], lm_head(final_norm(lv[L19][n, full_pos])).float(), gold_ng)
                _bucket(accs["base18"], lm_head(final_norm(h[n, full_pos])).float(), gold_ng)
                _bucket(accs["base19"], lm_head(final_norm(out.hidden_states[L19][n, full_pos])).float(), gold_ng)
                _bucket(accs["base20"], out.logits[n, full_pos].float(), gold_ng)
        loss2 = loss2 / N
        loss = loss + loss2_w * loss2
        agg["loss2"] /= N; agg["acc2"] /= N

    for k in ("loss1", "acc1"):
        agg[k] /= N
    return loss, agg, accs


@torch.no_grad()
def evaluate(model, head, val_records, tf, cfg, tap_index, embed, final_norm, lm_head, rotary_emb, dt, device,
             micro_bsz, max_batches, loss2_w=0.0, lora_mods=None):
    head.eval()
    agg = {"loss1": 0.0, "acc1": 0.0, "loss2": 0.0, "acc2": 0.0}
    merged = {}
    nb = 0
    for batch in batched(instance_stream(val_records, tf, shuffle=False, seed=0), micro_bsz):
        _, m, accs = run_micro_batch(model, head, collate(batch, device), cfg, tap_index, embed, final_norm,
                                     lm_head, rotary_emb, dt, device, loss1_w=1.0, loss2_w=loss2_w, val=True,
                                     lora_mods=lora_mods)
        for k in agg:
            agg[k] += m[k]
        for c, acc in accs.items():
            mc = merged.setdefault(c, _new_acc())
            for gi, (hh, cc, ll) in acc.items():
                a = mc[gi]; a[0] += hh; a[1] += cc; a[2] += ll
        nb += 1
        if nb >= max_batches:
            break
    head.train()
    r = {f"val_{k}": agg[k] / max(nb, 1) for k in agg}
    r["val_batches"] = nb
    r["pos"] = {c: _report(acc) for c, acc in merged.items()}    # curve -> {gen_idx: (acc, loss)}
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--gold_glob", required=True, help="JSONL shard glob, e.g. '../darc_gold/.../gold.*.jsonl'")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--tap_hidden_index", type=int, default=18,
                   help="hidden_states index to tap == probe-plot 'L{i}'; hs[i]=decoder layer (i-1) out")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--n_ar_passes", type=int, default=1, help="stacked AR passes (2 = pass1->h_ar1->pass2->h_ar2)")
    p.add_argument("--dynamic_k", action="store_true", help="pass-1 position-dependent top-k (5/10/25/50/100)")
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--micro_bsz", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--loss1_weight", type=float, default=1.0, help="AR tap-readout CE (trains attn+mlp)")
    p.add_argument("--loss2_weight", type=float, default=1.0,
                   help="inject->replay->final-output CE (trains the head/fuse); 0 = Loss-1 only")
    p.add_argument("--loss2_inject", choices=["ar_out", "soft", "ar_fuse"], default="ar_out",
                   help="ar_out = inject the AR residual output; ar_fuse = fuse(concat[h_ar, original h_18]) "
                        "(hidden-space, keeps sharp decode + rich hidden); soft = fuse on non-detached top-k prune")
    p.add_argument("--fuse_hidden_mult", type=int, default=6, help="'soft' fuse MLP: 2D -> mult*D -> D")
    p.add_argument("--lora_layers", default="", help="decoder layers to LoRA-adapt for the DARC replay, e.g. "
                   "'18' (2nd-last). '' = no LoRA (frozen replay). The last layer (19) is left frozen.")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_lr", type=float, default=1e-5, help="separate LR for the LoRA delta (pre-trained "
                   "layer -> smaller than the fresh head's --lr)")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=4000, help="ignored if --epochs > 0")
    p.add_argument("--epochs", type=float, default=0.0, help="if >0, run this many passes over the data")
    p.add_argument("--resume", action="store_true", help="resume from the latest head_step*.pt in out_dir")
    p.add_argument("--init_from", default=None,
                   help="partial-load head weights from a checkpoint (fresh step/opt/schedule) -- for A/B from "
                        "a shared Loss-1 head into different --loss2_inject / out_dir")
    p.add_argument("--keep_last", type=int, default=3, help="keep only the last N step checkpoints (+best)")
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_batches", type=int, default=30)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_records", type=int, default=0, help="cap records (smoke); 0 = all")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    mp = os.path.abspath(args.model_path)

    # ---- frozen backbone ----
    model = load_fused(mp, device)
    for pm in model.parameters():
        pm.requires_grad_(False)
    model.eval()
    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    final_norm = model.model.norm
    rotary_emb = model.model.rotary_emb
    dt = embed.weight.dtype                                            # bf16
    tap_index = args.tap_hidden_index                                 # direct hidden_states index (plot 'L{i}')

    cfg = DarcConfig(hidden_size=model.config.hidden_size, num_attention_heads=model.config.num_attention_heads,
                     num_key_value_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim,
                     intermediate_size=model.config.intermediate_size, vocab_size=model.config.vocab_size,
                     rotary_dim=getattr(model.config, "rotary_dim", 64), block_size=args.block_length,
                     tap_hidden_index=args.tap_hidden_index, top_k=args.top_k)
    setattr(cfg, "mask_token_id", MASK_ID)
    cfg.loss2_inject = args.loss2_inject                              # "ar_out" (no fuse) | "soft" (bigger fuse)
    cfg.n_ar_passes = args.n_ar_passes; cfg.dynamic_k = args.dynamic_k
    cfg.fuse_hidden_mult = args.fuse_hidden_mult
    head = DarcHead(cfg).to(device=device, dtype=torch.float32)        # fp32 master params; forward autocasts bf16
    head.train()

    # ---- optional LoRA on the DARC-replay decoder layer(s) (default: none) ----
    # The head refines hs[tap]; the FROZEN decoder layers above it can't consume that off-distribution hidden
    # (with-L18 > with-L19 -> the 2nd-last layer degrades it). LoRA-adapt those layer(s) so they learn to use
    # h_ar. The base metrics forward runs LoRA-OFF (honest frozen baseline); the DARC replay runs LoRA-ON.
    lora_mods = []
    lora_layer_idxs = [int(x) for x in args.lora_layers.split(",") if x.strip() != ""]
    if lora_layer_idxs:
        lora_mods = add_lora(model, lora_layer_idxs, r=args.lora_rank, alpha=args.lora_alpha)
        set_lora_enabled(lora_mods, False)                            # default OFF; run_micro_batch toggles it
    n_params = sum(p.numel() for p in head.parameters())
    n_lora = sum(p.numel() for p in lora_parameters(lora_mods))
    print(f"[train] head params={n_params/1e6:.1f}M  lora params={n_lora/1e6:.2f}M on layers={lora_layer_idxs} "
          f"tap_index={tap_index} dt={dt} block={args.block_length}")

    groups = [{"params": list(head.parameters()), "base_lr": args.lr}]
    if lora_mods:
        groups.append({"params": lora_parameters(lora_mods), "base_lr": args.lora_lr})
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    trainable = list(head.parameters()) + lora_parameters(lora_mods)  # for grad-clip

    def lr_factor(step):                                             # linear warmup -> cosine to 0.1x (per group)
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    # ---- data: read shards, deterministic rank-based train/val split ----
    records = list(iter_gold_records(args.gold_glob))
    if args.max_records:
        records = records[:args.max_records]
    vp = max(2, round(1.0 / args.val_frac))
    val_records = [r for r in records if int(r.get("rank", 0)) % vp == 0]
    train_records = [r for r in records if int(r.get("rank", 0)) % vp != 0]
    print(f"[train] records={len(records)} -> train={len(train_records)} val={len(val_records)} (1/{vp})")

    tf = lambda rec: process_darc_gold_example(rec, max_seq_len=args.max_seq_len, block_size=args.block_length,
                                               mode="reveal_prior")
    if args.epochs > 0:                                                # count instances (no model) -> max_steps
        n_inst = sum(len(tf(r)) for r in train_records)
        steps_per_epoch = max(1, math.ceil(n_inst / args.micro_bsz))
        args.max_steps = int(args.epochs * steps_per_epoch)
        print(f"[train] {n_inst} train instances -> {steps_per_epoch} steps/epoch -> max_steps={args.max_steps}")

    import glob as _glob
    def _ckpts():
        return sorted(_glob.glob(os.path.join(args.out_dir, "head_step*.pt")),
                      key=lambda p: int(p.split("head_step")[-1].split(".pt")[0]))

    best_acc = -1.0

    def save_ckpt(step, tag=None):
        name = f"head_{tag}.pt" if tag else f"head_step{step}.pt"
        path = os.path.join(args.out_dir, name)
        torch.save({"step": step, "config": vars(cfg), "state_dict": head.state_dict(),
                    "lora": lora_state_dict(lora_mods), "lora_layers": lora_layer_idxs,
                    "opt": opt.state_dict(), "best_acc": best_acc}, path)
        if tag is None and args.keep_last > 0:                        # prune old step checkpoints
            for old in _ckpts()[:-args.keep_last]:
                try:
                    os.remove(old)
                except OSError:
                    pass
        return path

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    logf = open(log_path, "a")

    def _load_head(sd):                                             # load only NAME+SHAPE-matching tensors
        msd = head.state_dict()                                     # (strict=False still errors on shape mismatch)
        keep = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
        head.load_state_dict(keep, strict=False)
        return len(keep), len(msd)

    def _load_lora(ck):                                             # load LoRA deltas if both ckpt+run have them
        if lora_mods and ck.get("lora"):
            nl = load_lora_state(lora_mods, ck["lora"])
            print(f"[train] loaded {nl} LoRA delta(s) for layers={lora_layer_idxs}")

    step = 0
    if args.init_from and not args.resume:                          # A/B: shared Loss-1 weights, fresh schedule
        ck = torch.load(args.init_from, map_location=device)
        nk, nt = _load_head(ck["state_dict"])
        _load_lora(ck)
        print(f"[train] init_from {args.init_from}: loaded {nk}/{nt} tensors (rest fresh: e.g. resized/new fuse); "
              f"fresh step/opt/schedule")
    if args.resume and _ckpts():
        ck = torch.load(_ckpts()[-1], map_location=device)
        nk, nt = _load_head(ck["state_dict"])
        _load_lora(ck)
        if nk == nt:
            try:
                if "opt" in ck:
                    opt.load_state_dict(ck["opt"])
            except (ValueError, KeyError, RuntimeError) as e:
                print(f"[train] resume: optimizer state not loaded ({e}); optimizer starts fresh")
        else:
            print(f"[train] resume PARTIAL head load ({nk}/{nt}); optimizer starts fresh")
        step = int(ck["step"]); best_acc = float(ck.get("best_acc", -1.0))
        print(f"[train] resumed from {_ckpts()[-1]} at step {step} (best_acc={best_acc:.3f})")

    t0 = time.time()
    running = {"loss1": 0.0, "acc1": 0.0, "loss2": 0.0, "acc2": 0.0, "k": 0}
    nonfinite = 0
    epoch = 0
    do2 = args.loss2_weight > 0
    while step < args.max_steps:
        stream = instance_stream(train_records, tf, shuffle=True, seed=args.seed + epoch)
        for batch in batched(stream, args.micro_bsz):
            if step >= args.max_steps:
                break
            _f = lr_factor(step)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * _f
            collect = ((step + 1) % args.log_every == 0)             # position-wise buckets only on log steps
            loss, m, accs = run_micro_batch(model, head, collate(batch, device), cfg, tap_index, embed, final_norm,
                                            lm_head, rotary_emb, dt, device,
                                            loss1_w=args.loss1_weight, loss2_w=args.loss2_weight,
                                            collect_pos=collect, lora_mods=lora_mods)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            if torch.isfinite(gn):                                    # bf16 grad-spike guard (skip the step)
                opt.step()
            else:
                nonfinite += 1
            opt.zero_grad(set_to_none=True)
            step += 1
            for kk in ("loss1", "acc1", "loss2", "acc2"):
                running[kk] += m[kk]
            running["k"] += 1

            if step % args.log_every == 0:
                k = running["k"]
                rec = {"step": step, "lr": args.lr * lr_factor(step), "loss1": running["loss1"] / k,
                       "acc1": running["acc1"] / k, "loss2": running["loss2"] / k, "acc2": running["acc2"] / k,
                       "grad_norm": float(gn), "ex_s": (step * args.micro_bsz) / (time.time() - t0)}
                extra = f" | loss2={rec['loss2']:.4f} acc2={rec['acc2']:.3f}" if do2 else ""
                print(f"[{step}/{args.max_steps}] loss1={rec['loss1']:.4f} acc1={rec['acc1']:.3f}{extra} "
                      f"lr={rec['lr']:.2e} gn={rec['grad_norm']:.2f} {rec['ex_s']:.1f}ex/s")
                logf.write(json.dumps({**rec, "split": "train"}) + "\n"); logf.flush()
                if accs.get("tap18"):                                 # position-wise acc @pos 1/4/8/16/32 (this batch)
                    r18 = _report(accs["tap18"])
                    print(f"        L18(=loss1) acc@pos1/4/8/16/32: {_fmt_pos(r18)}")
                    pos_rec = {"step": step, "split": "train_pos", "tap18": r18}
                    if accs.get("tap20"):
                        r20 = _report(accs["tap20"])
                        print(f"        L20(=loss2) acc@pos1/4/8/16/32: {_fmt_pos(r20)}")
                        pos_rec["tap20"] = r20
                    logf.write(json.dumps(pos_rec) + "\n"); logf.flush()
                running = {"loss1": 0.0, "acc1": 0.0, "loss2": 0.0, "acc2": 0.0, "k": 0}

            if args.eval_every and step % args.eval_every == 0:
                ev = evaluate(model, head, val_records, tf, cfg, tap_index, embed, final_norm, lm_head,
                              rotary_emb, dt, device, args.micro_bsz, args.eval_batches,
                              loss2_w=args.loss2_weight, lora_mods=lora_mods)
                sel = ev["val_acc2"] if do2 else ev["val_acc1"]      # best tracks the FINAL-output acc when Loss-2 on
                star = ""
                if sel > best_acc:
                    best_acc = sel; save_ckpt(step, tag="best"); star = " *best*"
                pos = ev.get("pos", {})                              # 6-way position-wise table (acc)
                print(f"[{step}] VAL acc @pos:    1     4     8    16    32       (no-tap=base logit-lens, "
                      f"with-tap=DARC){star} skips={nonfinite}")
                for c in ("base18", "tap18", "base19", "tap19", "base20", "tap20"):
                    if c in pos:
                        tag = {"base18": "no-tap L18", "tap18": "with   L18(=loss1)",
                               "base19": "no-tap L19", "tap19": "with   L19",
                               "base20": "no-tap L20", "tap20": "with   L20(=loss2)"}[c]
                        print(f"        {tag:18s} {_fmt_pos(pos[c])}")
                print(f"        [agg] val_acc1={ev['val_acc1']:.3f} val_acc2={ev['val_acc2']:.3f}")
                logf.write(json.dumps({"step": step, "split": "val", "best_acc": best_acc, **ev}) + "\n"); logf.flush()

            if args.save_every and step % args.save_every == 0:
                print(f"[{step}] saved {save_ckpt(step)}")
        epoch += 1

    # final eval + save
    ev = evaluate(model, head, val_records, tf, cfg, tap_index, embed, final_norm, lm_head,
                  rotary_emb, dt, device, args.micro_bsz, args.eval_batches,
                  loss2_w=args.loss2_weight, lora_mods=lora_mods)
    sel = ev["val_acc2"] if do2 else ev["val_acc1"]
    if sel > best_acc:
        best_acc = sel; save_ckpt(step, tag="best")
    save_ckpt(step, tag="final")
    logf.write(json.dumps({"step": step, "split": "val_final", "best_acc": best_acc, **ev}) + "\n"); logf.close()
    print(f"[done] {step} steps in {(time.time()-t0)/60:.1f}m  final VAL acc1={ev['val_acc1']:.3f} "
          f"best={best_acc:.3f}  nonfinite_skips={nonfinite}")


if __name__ == "__main__":
    main()
