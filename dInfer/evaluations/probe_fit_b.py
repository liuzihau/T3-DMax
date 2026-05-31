# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# PREMISE PROBE — Variant B, stage 2: block-level refiner fit + verdict.
#
# The per-position probe asked "is the converged token a function of THIS
# position's anchor alone?" -> no (~10% flip recovery). Variant B asks the fair
# question for T3-D: "given the static anchor + REVEALED NEIGHBORS, can a
# LIGHTWEIGHT block-level model recover the converged tokens?"
#
# We train a small transformer (a few layers) that takes, per block position,
# embed(current token) + proj(anchor), and predicts the converged block. Training
# = masked denoising: reveal a random fraction of decode positions with their
# CORRECT converged token, mask the rest, predict the masked ones. This is
# leak-free (targets are MASK in the input) and is exactly talk's job, distilled
# from the heavy model's converged output.
#
# Headline = flip_recovery vs reveal level:
#   - m=1.0 (no neighbors revealed)  ~ the per-position ceiling (sanity: ~10%).
#   - m=0.5 (half the neighbors)     = the joint-refinement test.
#   If flip_recovery rises sharply as neighbors are revealed, the neighbors carry
#   the joint info AND a lightweight model can use it -> v2's 0% is a TRAINING
#   problem, the mechanism is viable. If it stays flat/low, the static anchor is
#   bounded -> proceed to anchor-refresh.

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def split_by_group(group, val_frac, seed):
    g = torch.unique(group)
    gen = torch.Generator().manual_seed(seed)
    perm = g[torch.randperm(g.numel(), generator=gen)]
    n_val = max(1, int(round(val_frac * g.numel())))
    val_groups = set(perm[:n_val].tolist())
    is_val = torch.tensor([int(x) in val_groups for x in group.tolist()], dtype=torch.bool)
    return ~is_val, is_val


class AnchorRefiner(nn.Module):
    """Lightweight block-level refiner: embed(token) + proj(anchor) -> transformer
    -> converged token. Full self-attention within the block (mirrors talk)."""

    def __init__(self, vocab, d_model, d_anchor, block_len, n_layers, n_heads):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab, d_model)
        self.anchor_proj = nn.Linear(d_anchor, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, block_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, input_ids, anchor):
        h = self.tok_embed(input_ids) + self.anchor_proj(anchor) + self.pos_embed
        h = self.encoder(h)
        return self.head(h)


def main():
    p = argparse.ArgumentParser(description="Premise probe Variant B — block refiner fit + verdict")
    p.add_argument("--data", required=True, help="probe_collect_blocks.py .pt output")
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    blob = torch.load(args.data, map_location="cpu")
    anchor = blob["anchor"].float()                  # [Nb, blk, D]
    y = blob["y"]                                     # [Nb, blk]
    arg0 = blob["arg0"]                               # [Nb, blk]
    decode_mask = blob["decode_mask"]                # [Nb, blk] bool
    group = blob["group"]                            # [Nb]
    mask_id = int(blob["mask_id"])
    meta = blob.get("meta", {})
    Nb, blk, D = anchor.shape
    print(f"[probe-b] {Nb} blocks  block_len={blk}  D={D}  from {meta.get('model_path','?')}")

    # Compact shared vocab (input tokens + MASK + targets all live in y ∪ {mask}).
    uniq = torch.unique(torch.cat([y.reshape(-1), torch.tensor([mask_id])]))
    tok2id = {int(t): i for i, t in enumerate(uniq.tolist())}
    V = len(uniq)
    MASKc = tok2id[mask_id]
    yc = torch.tensor([[tok2id[int(t)] for t in row] for row in y.tolist()], dtype=torch.long)
    arg0c = torch.tensor([[tok2id.get(int(t), MASKc) for t in row] for row in arg0.tolist()], dtype=torch.long)
    print(f"[probe-b] compact vocab={V}")

    tr, va = split_by_group(group, args.val_frac, args.seed)
    dev = args.device
    A_tr, A_va = anchor[tr].to(dev), anchor[va].to(dev)
    y_tr, y_va = yc[tr].to(dev), yc[va].to(dev)
    dm_tr, dm_va = decode_mask[tr].to(dev), decode_mask[va].to(dev)
    arg0_va = arg0c[va].to(dev)
    print(f"[probe-b] train={int(tr.sum())} blocks  val={int(va.sum())} blocks  "
          f"({len(torch.unique(group[tr]))}/{len(torch.unique(group[va]))} prompts)")

    model = AnchorRefiner(V, args.d_model, D, blk, args.n_layers, args.n_heads).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    n = A_tr.shape[0]
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    def make_input(y_blocks, dm_blocks, m):
        """Mask each decode position with prob m (m=None -> per-block U(0.1,0.95))."""
        B = y_blocks.shape[0]
        if m is None:
            rho = torch.rand(B, 1, device=dev, generator=gen) * 0.85 + 0.10
        else:
            rho = torch.full((B, 1), float(m), device=dev)
        mask_sel = (torch.rand(B, blk, device=dev, generator=gen) < rho) & dm_blocks
        inp = y_blocks.clone()
        inp[mask_sel] = MASKc
        return inp, mask_sel

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=dev)
        tot, seen = 0.0, 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            inp, mask_sel = make_input(y_tr[idx], dm_tr[idx], None)
            logits = model(inp, A_tr[idx])
            tgt = y_tr[idx].clone()
            tgt[~mask_sel] = -100
            loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * idx.numel(); seen += idx.numel()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{args.epochs}  train_ce={tot/max(seen,1):.4f}")

    # ---- eval at fixed reveal levels ----
    model.eval()
    print("\n" + "=" * 72)
    print("PREMISE PROBE — VARIANT B RESULT  (acc / flip_recovery on masked decode positions)")
    print("=" * 72)
    print(f"{'mask m':>8} {'revealed':>9} {'acc':>8} {'flip_recovery':>15}")
    results = {}
    eval_gen = torch.Generator(device=dev).manual_seed(args.seed + 1)
    with torch.no_grad():
        for m in [1.0, 0.75, 0.5, 0.25]:
            B = A_va.shape[0]
            rho = torch.full((B, 1), m, device=dev)
            mask_sel = (torch.rand(B, blk, device=dev, generator=eval_gen) < rho) & dm_va
            inp = y_va.clone(); inp[mask_sel] = MASKc
            pred = model(inp, A_va).argmax(dim=-1)
            correct = (pred == y_va) & mask_sel
            acc = correct.sum().item() / max(mask_sel.sum().item(), 1)
            flip = mask_sel & (arg0_va != y_va)
            flip_rec = ((pred == y_va) & flip).sum().item() / max(flip.sum().item(), 1)
            results[m] = (acc, flip_rec)
            print(f"{m:>8.2f} {1-m:>9.2f} {acc:>8.1%} {flip_rec:>15.1%}")
    print("=" * 72)

    # ---- verdict ----
    fr_half = results[0.5][1]
    fr_none = results[1.0][1]
    print("VERDICT  (compare to per-position flip_recovery ~10%)")
    print(f"  anchor-only (m=1.0) flip_recovery = {fr_none:.1%}   "
          f"half-revealed (m=0.5) flip_recovery = {fr_half:.1%}")
    if fr_half >= 0.55:
        print("  HIGH: a lightweight block-level model + anchor + revealed neighbors RECOVERS the")
        print("  joint refinement. The mechanism is viable -> v2's 0% is a TRAINING/instantiation")
        print("  problem, not an architectural bound. Fix training (and the DMax backbone swap")
        print("  helps the anchor) and retry. Anchor-refresh may not be needed.")
    elif fr_half <= 0.25:
        print("  LOW: even with correct neighbors revealed, the lightweight model can't recover the")
        print("  flips from the STATIC anchor. The think-once anchor is fundamentally bounded ->")
        print("  proceed to step 2: anchor-refresh every M iters (re-run think to re-inject reveals).")
    else:
        print("  MIDDLING: neighbors help (vs ~10% per-position) but a static-anchor lightweight model")
        print("  leaves accuracy on the table. Anchor-refresh likely needed for the hard flips;")
        print("  worth prototyping it (step 2) and comparing.")
    print("=" * 72)


if __name__ == "__main__":
    main()
