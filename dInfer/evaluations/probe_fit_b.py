# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# PREMISE PROBE — Variant B, stage 2: block-level refiner fit + verdict.
#
# Asks the fair question for T3-D: given the static anchor + REVEALED NEIGHBORS,
# can a LIGHTWEIGHT block-level model recover the converged tokens (the flips)?
#
# We train a small transformer that takes embed(current token) [+ proj(anchor)]
# and predicts the converged block via masked denoising: reveal a random fraction
# of decode positions with their CORRECT converged token, mask the rest, predict
# the masked ones (leak-free; targets are MASK in the input).
#
# CRITICAL CONTROL: we train TWO models -- with-anchor and NO-anchor (tokens
# only). The no-anchor model is the sanity check + the real answer:
#   - if no-anchor acc RISES with reveal, the pipeline genuinely uses neighbors;
#   - its flip_recovery vs the per-position anchor-only baseline (~10%) tells us
#     whether the flips live in within-block neighbors at all.
#   The with-anchor model is prone to "anchor laziness" (ignoring the variably-
#   present reveals because the anchor is always there) -> flat acc across reveal
#   is that artifact, NOT evidence that the static anchor is bounded.

import argparse

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
    """Lightweight block-level refiner. use_anchor=False -> tokens-only control."""

    def __init__(self, vocab, d_model, d_anchor, block_len, n_layers, n_heads, use_anchor=True):
        super().__init__()
        self.use_anchor = use_anchor
        self.tok_embed = nn.Embedding(vocab, d_model)
        self.anchor_proj = nn.Linear(d_anchor, d_model) if use_anchor else None
        self.pos_embed = nn.Parameter(torch.zeros(1, block_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, input_ids, anchor):
        h = self.tok_embed(input_ids) + self.pos_embed
        if self.use_anchor:
            h = h + self.anchor_proj(anchor)
        return self.head(self.encoder(h))


def main():
    p = argparse.ArgumentParser(description="Premise probe Variant B — block refiner + no-anchor control")
    p.add_argument("--data", required=True)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    blob = torch.load(args.data, map_location="cpu")
    anchor = blob["anchor"].float()
    y = blob["y"]; arg0 = blob["arg0"]; decode_mask = blob["decode_mask"]; group = blob["group"]
    mask_id = int(blob["mask_id"]); meta = blob.get("meta", {})
    Nb, blk, D = anchor.shape
    print(f"[probe-b] {Nb} blocks  block_len={blk}  D={D}  from {meta.get('model_path','?')}")

    uniq = torch.unique(torch.cat([y.reshape(-1), torch.tensor([mask_id])]))
    tok2id = {int(t): i for i, t in enumerate(uniq.tolist())}
    V = len(uniq); MASKc = tok2id[mask_id]
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

    EVAL_MS = [1.0, 0.75, 0.5, 0.25]
    eval_gen = torch.Generator(device=dev).manual_seed(args.seed + 1)
    eval_masks = {}   # m -> mask_sel for val blocks (shared across both models)
    Bv = A_va.shape[0]
    for m in EVAL_MS:
        rho = torch.full((Bv, 1), m, device=dev)
        eval_masks[m] = (torch.rand(Bv, blk, device=dev, generator=eval_gen) < rho) & dm_va

    def train_model(use_anchor):
        torch.manual_seed(args.seed)
        model = AnchorRefiner(V, args.d_model, D, blk, args.n_layers, args.n_heads, use_anchor).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        gen = torch.Generator(device=dev).manual_seed(args.seed + (0 if use_anchor else 7))
        n = A_tr.shape[0]
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, args.batch_size):
                idx = perm[i:i + args.batch_size]
                rho = torch.rand(idx.numel(), 1, device=dev, generator=gen) * 0.85 + 0.10
                mask_sel = (torch.rand(idx.numel(), blk, device=dev, generator=gen) < rho) & dm_tr[idx]
                inp = y_tr[idx].clone(); inp[mask_sel] = MASKc
                logits = model(inp, A_tr[idx])
                tgt = y_tr[idx].clone(); tgt[~mask_sel] = -100
                loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=-100)
                opt.zero_grad(); loss.backward(); opt.step()
            if (ep + 1) % 20 == 0:
                model.eval()
                with torch.no_grad():
                    ms = eval_masks[0.5]
                    inp = y_va.clone(); inp[ms] = MASKc
                    pred = model(inp, A_va).argmax(-1)
                    acc = ((pred == y_va) & ms).sum().item() / max(ms.sum().item(), 1)
                print(f"    [{'anchor' if use_anchor else 'tokens-only'}] epoch {ep+1}/{args.epochs}  val_acc@m0.5={acc:.3f}")
        return model

    @torch.no_grad()
    def eval_curve(model, label):
        model.eval()
        print(f"\n  {label}:   {'mask m':>7} {'revealed':>9} {'acc':>8} {'flip_recovery':>14}")
        out = {}
        for m in EVAL_MS:
            ms = eval_masks[m]
            inp = y_va.clone(); inp[ms] = MASKc
            pred = model(inp, A_va).argmax(-1)
            acc = ((pred == y_va) & ms).sum().item() / max(ms.sum().item(), 1)
            flip = ms & (arg0_va != y_va)
            fr = ((pred == y_va) & flip).sum().item() / max(flip.sum().item(), 1)
            out[m] = (acc, fr)
            print(f"  {'':>{len(label)}}    {m:>7.2f} {1-m:>9.2f} {acc:>8.1%} {fr:>14.1%}")
        return out

    print("\n[probe-b] training WITH-anchor model ...")
    m_anchor = train_model(True)
    print("[probe-b] training NO-anchor (tokens-only) CONTROL ...")
    m_tokens = train_model(False)

    print("\n" + "=" * 76)
    print("PREMISE PROBE — VARIANT B  (per-position anchor-only baseline flip_recovery ~10%)")
    print("=" * 76)
    res_a = eval_curve(m_anchor, "with-anchor")
    res_t = eval_curve(m_tokens, "tokens-only")
    print("=" * 76)

    # ---- verdict (centered on the tokens-only control) ----
    na_acc_lift = res_t[0.25][0] - res_t[1.0][0]     # does revealing neighbors raise acc?
    na_flip_hi = res_t[0.25][1]                       # flips recoverable with 75% neighbors?
    a_acc_lift = res_a[0.25][0] - res_a[1.0][0]
    print("VERDICT")
    print(f"  tokens-only acc lift (m=1.0 -> 0.25): {na_acc_lift:+.1%}   "
          f"with-anchor acc lift: {a_acc_lift:+.1%}")
    if na_acc_lift < 0.05:
        print("  ⚠️ PIPELINE WARNING: even the tokens-only control barely uses revealed neighbors")
        print("  (acc ~flat across reveal). The masking/attention path is suspect — investigate the")
        print("  probe before drawing ANY conclusion about the anchor.")
    elif na_flip_hi >= 0.40:
        print(f"  NEIGHBORS CARRY THE FLIPS: tokens-only recovers {na_flip_hi:.0%} of flips at 75%")
        print("  reveal (vs ~10% anchor-only per-position). So the flips ARE within-block recoverable")
        print("  -> a properly-trained talk CAN get them -> v2's 0% is a TRAINING/instantiation")
        print("  problem, not a static-anchor bound. Do NOT pivot to anchor-refresh yet; fix talk.")
        if a_acc_lift < 0.05:
            print("  (The with-anchor model went anchor-lazy — flat acc — so ignore its flatness.)")
    elif na_flip_hi <= 0.20:
        print(f"  FLIPS NOT IN NEIGHBORS: even tokens-only with 75% of the correct block revealed")
        print(f"  recovers only {na_flip_hi:.0%} of flips. They need cross-block / global info the")
        print("  static block lacks -> the think-once anchor is genuinely bounded -> anchor-refresh.")
    else:
        print(f"  MIDDLING: tokens-only recovers {na_flip_hi:.0%} of flips at 75% reveal. Neighbors")
        print("  help partially; anchor-refresh likely still needed for the hard flips. Prototype it.")
    print("=" * 76)


if __name__ == "__main__":
    main()
