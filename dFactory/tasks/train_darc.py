# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# DARC first-trial trainer (SINGLE GPU, plain PyTorch). Trains the ~50M DARC head on a FROZEN DMax-16B using
# the reveal-prior gold data. Reuses the exact components validated by dFactory/scripts/smoke_darc_integration.py
# (load_fused, tap hidden_states[tap_layer+1], build_block_causal_mask, DarcHead) so training == the smoke.
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


def run_micro_batch(model, head, batch, cfg, tap_index, embed, final_norm, lm_head, rotary_emb, dt, device):
    """Batched frozen heavy forward on the reveal-prior noisy stream -> tap -> per-instance head on the active
    block. Returns (loss_tensor, metrics_dict)."""
    noisy, labels, act = batch["noisy"], batch["labels"], batch["active_bs"]
    N, L = noisy.shape
    B = cfg.block_size
    with torch.no_grad():                                              # frozen backbone
        attn = build_block_causal_mask(L, B, dtype=dt, device=device)  # [1,1,L,L]
        attn = attn.expand(N, *attn.shape[1:])                         # model needs a per-batch (N,1,L,L) mask
        pos = torch.arange(L, device=device)[None].expand(N, L)
        out = model(input_ids=noisy, attention_mask=attn, position_ids=pos,
                    use_cache=False, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[tap_index]                              # [N,L,D]
        cos, sin = rotary_emb(h, pos)                                 # [N,L,rot]
    loss = 0.0
    agg = {"loss1": 0.0, "acc1": 0.0, "n_sup": 0}
    for n in range(N):
        bs = int(act[n]); be = bs + B
        with torch.autocast(device_type="cuda", dtype=dt):
            l, m = head.forward_train(h[n:n + 1, bs:be], noisy[n:n + 1, bs:be], labels[n:n + 1, bs:be],
                                      cos[n:n + 1, bs:be], sin[n:n + 1, bs:be], embed, final_norm, lm_head)
        loss = loss + l
        agg["loss1"] += m["loss1"]; agg["acc1"] += m["acc1"]; agg["n_sup"] += m["n_sup"]
    loss = loss / N
    for k in ("loss1", "acc1"):
        agg[k] /= N
    return loss, agg


@torch.no_grad()
def evaluate(model, head, val_records, tf, cfg, tap_index, embed, final_norm, lm_head, rotary_emb, dt, device,
             micro_bsz, max_batches):
    head.eval()
    tl, ta, nb = 0.0, 0.0, 0
    for batch in batched(instance_stream(val_records, tf, shuffle=False, seed=0), micro_bsz):
        _, m = run_micro_batch(model, head, collate(batch, device), cfg, tap_index, embed, final_norm,
                               lm_head, rotary_emb, dt, device)
        tl += m["loss1"]; ta += m["acc1"]; nb += 1
        if nb >= max_batches:
            break
    head.train()
    return {"val_loss": tl / max(nb, 1), "val_acc1": ta / max(nb, 1), "val_batches": nb}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--gold_glob", required=True, help="JSONL shard glob, e.g. '../darc_gold/.../gold.*.jsonl'")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--tap_layer", type=int, default=18)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--micro_bsz", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=4000)
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
    tap_index = args.tap_layer + 1

    cfg = DarcConfig(hidden_size=model.config.hidden_size, num_attention_heads=model.config.num_attention_heads,
                     num_key_value_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim,
                     intermediate_size=model.config.intermediate_size, vocab_size=model.config.vocab_size,
                     rotary_dim=getattr(model.config, "rotary_dim", 64), block_size=args.block_length,
                     tap_layer=args.tap_layer, top_k=args.top_k)
    setattr(cfg, "mask_token_id", MASK_ID)
    head = DarcHead(cfg).to(device=device, dtype=torch.float32)        # fp32 master params; forward autocasts bf16
    head.train()
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[train] head params={n_params/1e6:.1f}M  tap_index={tap_index} dt={dt} block={args.block_length}")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    def lr_at(step):                                                  # linear warmup -> cosine to 0.1*lr
        if step < args.warmup_steps:
            return args.lr * step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))

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
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    logf = open(log_path, "a")

    step = 0
    t0 = time.time()
    running = {"loss1": 0.0, "acc1": 0.0, "k": 0}
    epoch = 0
    while step < args.max_steps:
        stream = instance_stream(train_records, tf, shuffle=True, seed=args.seed + epoch)
        for batch in batched(stream, args.micro_bsz):
            if step >= args.max_steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss, m = run_micro_batch(model, head, collate(batch, device), cfg, tap_index, embed, final_norm,
                                      lm_head, rotary_emb, dt, device)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            running["loss1"] += m["loss1"]; running["acc1"] += m["acc1"]; running["k"] += 1

            if step % args.log_every == 0:
                k = running["k"]
                rec = {"step": step, "lr": lr_at(step), "loss": running["loss1"] / k,
                       "acc1": running["acc1"] / k, "grad_norm": float(gn),
                       "ex_s": (step * args.micro_bsz) / (time.time() - t0)}
                print(f"[{step}/{args.max_steps}] loss={rec['loss']:.4f} acc1={rec['acc1']:.3f} "
                      f"lr={rec['lr']:.2e} gn={rec['grad_norm']:.2f} {rec['ex_s']:.1f}ex/s")
                logf.write(json.dumps({**rec, "split": "train"}) + "\n"); logf.flush()
                running = {"loss1": 0.0, "acc1": 0.0, "k": 0}

            if args.eval_every and step % args.eval_every == 0:
                ev = evaluate(model, head, val_records, tf, cfg, tap_index, embed, final_norm, lm_head,
                              rotary_emb, dt, device, args.micro_bsz, args.eval_batches)
                print(f"[{step}] VAL loss={ev['val_loss']:.4f} acc1={ev['val_acc1']:.3f} "
                      f"(baseline probe ~0.23 uncond, ~0.76 clean-prefix ceiling)")
                logf.write(json.dumps({"step": step, "split": "val", **ev}) + "\n"); logf.flush()

            if args.save_every and step % args.save_every == 0:
                ckpt = os.path.join(args.out_dir, f"head_step{step}.pt")
                torch.save({"step": step, "config": vars(cfg), "state_dict": head.state_dict()}, ckpt)
                print(f"[{step}] saved {ckpt}")
        epoch += 1

    # final eval + save
    ev = evaluate(model, head, val_records, tf, cfg, tap_index, embed, final_norm, lm_head,
                  rotary_emb, dt, device, args.micro_bsz, args.eval_batches)
    torch.save({"step": step, "config": vars(cfg), "state_dict": head.state_dict()},
               os.path.join(args.out_dir, "head_final.pt"))
    logf.write(json.dumps({"step": step, "split": "val_final", **ev}) + "\n"); logf.close()
    print(f"[done] {step} steps in {(time.time()-t0)/60:.1f}m  final VAL acc1={ev['val_acc1']:.3f}")


if __name__ == "__main__":
    main()
