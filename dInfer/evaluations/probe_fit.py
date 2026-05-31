# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# PREMISE PROBE — stage 2: fit + verdict.
#
# Loads the (h0, iter0_argmax, y_converged, group) dataset from probe_collect.py
# and asks: can a LIGHTWEIGHT head recover the converged token from the iter-0
# hidden state alone? Reports the numbers that decide whether T3-D's "1 heavy
# forward + N lightweight refines" split is achievable on this backbone.
#
# Metrics (all on a prompt-disjoint validation split):
#   flip_rate        = P(iter0_argmax != y_converged)
#                      how much the heavy iterative decode CHANGES vs its one-shot.
#                      This is the headroom; if ~0 there is nothing to refine.
#   acc_iter0        = P(iter0_argmax == y) — the frozen lm_head one-shot (do-nothing baseline).
#   acc_linear/_mlp  = P(probe == y) for a trained linear / MLP head on h0.
#   flip_recovery    = on flip positions only, P(probe == y).
#                      THE headline number: high => iter-0 hidden ALREADY encodes
#                      the converged answer (premise holds, a light refiner suffices);
#                      low => the answer needs info only later heavy forwards add.
#
# Heads are deliberately tiny (a linear readout, and a 2-layer MLP ~ delta_head
# scale) over a CLOSED label space (tokens observed as converged targets), so
# training is seconds-to-minutes and the result is purely "is the info linearly/
# shallow-nonlinearly present in h0".

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def split_by_group(group, val_frac, seed):
    """Prompt-disjoint split: whole prompts go to train or val (no position leakage)."""
    g = torch.unique(group)
    gen = torch.Generator().manual_seed(seed)
    perm = g[torch.randperm(g.numel(), generator=gen)]
    n_val = max(1, int(round(val_frac * g.numel())))
    val_groups = set(perm[:n_val].tolist())
    is_val = torch.tensor([int(x) in val_groups for x in group.tolist()], dtype=torch.bool)
    return ~is_val, is_val


class MLPHead(nn.Module):
    def __init__(self, d_in, n_out, hidden_mult=4):
        super().__init__()
        h = d_in * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, n_out),
        )

    def forward(self, x):
        return self.net(x)


def train_head(head, Xtr, ytr, Xva, yva, epochs, bs, lr, device, label="head"):
    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)
    n = Xtr.shape[0]
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            logits = head(Xtr[idx])
            loss = F.cross_entropy(logits, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * idx.numel()
        head.eval()
        with torch.no_grad():
            va_pred = head(Xva).argmax(dim=-1)
            va_acc = (va_pred == yva).float().mean().item()
        print(f"  [{label}] epoch {ep+1}/{epochs}  train_ce={tot/n:.4f}  val_acc={va_acc:.4f}")
    return head


def main():
    p = argparse.ArgumentParser(description="Premise probe — fit lightweight head + verdict")
    p.add_argument("--data", required=True, help="probe_collect.py .pt output")
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mlp_hidden_mult", type=int, default=4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    blob = torch.load(args.data, map_location="cpu")
    H0 = blob["h0"].float()                 # [N, D]
    ARG0 = blob["iter0_argmax"]             # [N]
    Y = blob["y_converged"]                 # [N]
    GROUP = blob["group"]                   # [N]
    meta = blob.get("meta", {})
    N, D = H0.shape
    print(f"[probe-fit] N={N} positions  D={D}  from {meta.get('model_path','?')}")

    # Closed label space = tokens that appear as converged targets.
    uniq = torch.unique(Y)
    tok2lab = {int(t): i for i, t in enumerate(uniq.tolist())}
    n_out = len(uniq)
    Ylab = torch.tensor([tok2lab[int(t)] for t in Y.tolist()], dtype=torch.long)
    inv = uniq.clone()                      # compact idx -> original token id
    print(f"[probe-fit] closed label space: {n_out} distinct converged tokens")

    tr, va = split_by_group(GROUP, args.val_frac, args.seed)
    dev = args.device
    Xtr, Xva = H0[tr].to(dev), H0[va].to(dev)
    ytr, yva = Ylab[tr].to(dev), Ylab[va].to(dev)
    arg0_va = ARG0[va].to(dev)
    y_tok_va = Y[va].to(dev)
    inv_dev = inv.to(dev)
    print(f"[probe-fit] train={int(tr.sum())}  val={int(va.sum())}  "
          f"(prompt-disjoint: {len(torch.unique(GROUP[tr]))} / {len(torch.unique(GROUP[va]))} prompts)")

    # ---- baselines + heads ------------------------------------------------
    flip_mask = (arg0_va != y_tok_va)
    flip_rate = flip_mask.float().mean().item()
    acc_iter0 = (arg0_va == y_tok_va).float().mean().item()

    print("\n[probe-fit] training LINEAR head ...")
    lin = train_head(nn.Linear(D, n_out), Xtr, ytr, Xva, yva,
                     args.epochs, args.batch_size, args.lr, dev, label="linear")
    print("[probe-fit] training MLP head ...")
    mlp = train_head(MLPHead(D, n_out, args.mlp_hidden_mult), Xtr, ytr, Xva, yva,
                     args.epochs, args.batch_size, args.lr, dev, label="mlp")

    def eval_head(head):
        with torch.no_grad():
            pred_tok = inv_dev[head(Xva).argmax(dim=-1)]
        acc = (pred_tok == y_tok_va).float().mean().item()
        if flip_mask.any():
            flip_rec = (pred_tok[flip_mask] == y_tok_va[flip_mask]).float().mean().item()
        else:
            flip_rec = float("nan")
        return acc, flip_rec

    acc_lin, rec_lin = eval_head(lin)
    acc_mlp, rec_mlp = eval_head(mlp)

    # ---- report -----------------------------------------------------------
    print("\n" + "=" * 72)
    print("PREMISE PROBE — RESULT")
    print("=" * 72)
    print(f"flip_rate  (iter0 != converged)         : {flip_rate:.1%}   "
          f"<- headroom iteration buys on this backbone")
    print(f"acc_iter0  (frozen lm_head one-shot)     : {acc_iter0:.1%}   <- do-nothing baseline")
    print(f"acc_linear (trained linear on h0)        : {acc_lin:.1%}")
    print(f"acc_mlp    (trained 2-layer MLP on h0)   : {acc_mlp:.1%}")
    print(f"flip_recovery_linear                     : {rec_lin:.1%}")
    print(f"flip_recovery_mlp                        : {rec_mlp:.1%}   <- HEADLINE")
    print("=" * 72)

    # ---- verdict ----------------------------------------------------------
    print("VERDICT")
    if flip_rate < 0.10:
        print(f"  flip_rate is low ({flip_rate:.1%}): the heavy iterative decode barely changes")
        print("  its one-shot answer. Refinement value is small -> talk mostly has to COPY the")
        print("  anchor, which is easy. Premise holds trivially; T3-D should match the backbone")
        print("  if talk learns to copy. (Re-check on harder data if you want more headroom.)")
    elif rec_mlp >= 0.60:
        print(f"  flip_recovery_mlp is HIGH ({rec_mlp:.1%}) at flip_rate {flip_rate:.1%}: the iter-0")
        print("  hidden state ALREADY encodes the converged answer, recoverable by a lightweight")
        print("  head. This is the strong form of the hypothesis -> the '1 heavy + N light' split")
        print("  is achievable. Proceed: swap backbone to DMax + train talk as the refiner.")
    elif rec_mlp <= 0.30:
        print(f"  flip_recovery_mlp is LOW ({rec_mlp:.1%}) at flip_rate {flip_rate:.1%}: the iter-0")
        print("  hidden does NOT contain the converged answer in a shallow-recoverable form. The")
        print("  refinement needs information only later HEAVY forwards add (joint conditioning on")
        print("  revealed neighbors). Per-position lightweight refine is bounded -> pivot:")
        print("   - refresh the anchor every M iters (re-run think periodically; still a win), or")
        print("   - give talk neighbor conditioning and re-run this probe as Variant B (block-level).")
    else:
        print(f"  flip_recovery_mlp is MIDDLING ({rec_mlp:.1%}) at flip_rate {flip_rate:.1%}: partial")
        print("  signal in h0. Worth trying T3-D, but expect the lightweight talk to leave some")
        print("  accuracy on the table vs the full backbone. Consider Variant B (neighbor-")
        print("  conditioned probe) and/or periodic anchor refresh before committing the long run.")
    print("=" * 72)


if __name__ == "__main__":
    main()
