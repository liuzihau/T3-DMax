# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# PREMISE PROBE — Variant B, stage 2: block-level refiner fit + verdict.
#
# Asks the fair question for T3-D: given the static anchor + REVEALED NEIGHBORS,
# can a LIGHTWEIGHT block-level model recover the converged tokens (the flips)?
# We train a small transformer (masked denoising of the converged block,
# conditioned on the anchor), and a NO-anchor control (tokens only).
#
# DIAGNOSTIC INSTRUMENTATION: we report per-epoch TRAIN ce + TRAIN acc, and a
# side-by-side TRAIN-vs-VAL acc / flip_recovery table at each reveal level. The
# key tell:
#   - TRAIN acc RISES with reveal but VAL flat  -> model DOES learn to use
#     neighbors, just overfits (1222 blocks) -> the fix is MORE DATA, and it
#     means neighbors genuinely carry the flips (relevant for T3-D).
#   - TRAIN acc ALSO flat with reveal           -> model ignores neighbors even
#     on train (anchor is a sufficient shortcut, or tokens-only can't learn the
#     neighbor function from this little data) -> deeper issue, not just data.

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


EVAL_MS = [1.0, 0.75, 0.5, 0.25]


def build_fixed_masks(dm, blk, dev, seed):
    """Per reveal-level m, a fixed bernoulli(m) mask over decode positions."""
    g = torch.Generator(device=dev).manual_seed(seed)
    B = dm.shape[0]
    masks = {}
    for m in EVAL_MS:
        masks[m] = (torch.rand(B, blk, device=dev, generator=g) < m) & dm
    return masks


def main():
    p = argparse.ArgumentParser(description="Premise probe Variant B — fit + train/val diagnostics")
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
    arg0_tr, arg0_va = arg0c[tr].to(dev), arg0c[va].to(dev)
    print(f"[probe-b] train={int(tr.sum())} blocks  val={int(va.sum())} blocks  "
          f"({len(torch.unique(group[tr]))}/{len(torch.unique(group[va]))} prompts)")

    masks_tr = build_fixed_masks(dm_tr, blk, dev, args.seed + 2)
    masks_va = build_fixed_masks(dm_va, blk, dev, args.seed + 1)

    @torch.no_grad()
    def metrics(model, A, y_, arg0_, masks):
        """Returns {m: (acc, flip_recovery)} over the given dataset + fixed masks."""
        model.eval()
        out = {}
        for m in EVAL_MS:
            ms = masks[m]
            inp = y_.clone(); inp[ms] = MASKc
            pred = model(inp, A).argmax(-1)
            corr = (pred == y_) & ms
            acc = corr.sum().item() / max(ms.sum().item(), 1)
            flip = ms & (arg0_ != y_)
            fr = ((pred == y_) & flip).sum().item() / max(flip.sum().item(), 1)
            out[m] = (acc, fr)
        return out

    def train_model(use_anchor):
        torch.manual_seed(args.seed)
        model = AnchorRefiner(V, args.d_model, D, blk, args.n_layers, args.n_heads, use_anchor).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        gen = torch.Generator(device=dev).manual_seed(args.seed + (0 if use_anchor else 7))
        n = A_tr.shape[0]
        tag = "anchor" if use_anchor else "tokens"
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(n, device=dev)
            ce_sum, corr_sum, tok_sum = 0.0, 0, 0
            for i in range(0, n, args.batch_size):
                idx = perm[i:i + args.batch_size]
                rho = torch.rand(idx.numel(), 1, device=dev, generator=gen) * 0.85 + 0.10
                mask_sel = (torch.rand(idx.numel(), blk, device=dev, generator=gen) < rho) & dm_tr[idx]
                inp = y_tr[idx].clone(); inp[mask_sel] = MASKc
                logits = model(inp, A_tr[idx])
                tgt = y_tr[idx].clone(); tgt[~mask_sel] = -100
                loss = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=-100)
                opt.zero_grad(); loss.backward(); opt.step()
                nmask = int(mask_sel.sum().item())
                ce_sum += loss.item() * nmask; tok_sum += nmask
                corr_sum += int(((logits.argmax(-1) == y_tr[idx]) & mask_sel).sum().item())
            if (ep + 1) % 10 == 0 or ep == 0:
                tr_ce = ce_sum / max(tok_sum, 1)
                tr_acc = corr_sum / max(tok_sum, 1)
                print(f"    [{tag}] epoch {ep+1:>3}/{args.epochs}  train_ce={tr_ce:.3f}  "
                      f"train_acc@randmask={tr_acc:.3f}")
        return model

    def print_table(model, label):
        mt = metrics(model, A_tr, y_tr, arg0_tr, masks_tr)
        mv = metrics(model, A_va, y_va, arg0_va, masks_va)
        print(f"\n  {label}:")
        print(f"    {'m':>5} {'revealed':>9} {'TRAIN_acc':>10} {'VAL_acc':>9} {'TRAIN_flipR':>12} {'VAL_flipR':>10}")
        for m in EVAL_MS:
            print(f"    {m:>5.2f} {1-m:>9.2f} {mt[m][0]:>10.1%} {mv[m][0]:>9.1%} "
                  f"{mt[m][1]:>12.1%} {mv[m][1]:>10.1%}")
        return mt, mv

    print("\n[probe-b] training WITH-anchor model ...")
    m_anchor = train_model(True)
    print("[probe-b] training NO-anchor (tokens-only) CONTROL ...")
    m_tokens = train_model(False)

    print("\n" + "=" * 78)
    print("VARIANT B — TRAIN vs VAL  (per-position anchor-only baseline flip_recovery ~10%)")
    print("=" * 78)
    at_tr, at_va = print_table(m_anchor, "with-anchor")
    tt_tr, tt_va = print_table(m_tokens, "tokens-only")
    print("=" * 78)

    # ---- verdict using the TRAIN-vs-reveal tell ----
    def lift(d):
        return d[0.25][0] - d[1.0][0]          # acc gain from revealing 75% vs 0%
    print("VERDICT")
    print(f"  TRAIN acc lift (m1.0->0.25): anchor={lift(at_tr):+.1%}  tokens-only={lift(tt_tr):+.1%}")
    print(f"  VAL   acc lift (m1.0->0.25): anchor={lift(at_va):+.1%}  tokens-only={lift(tt_va):+.1%}")
    tok_train_uses_nbrs = lift(tt_tr) >= 0.10
    tok_val_uses_nbrs = lift(tt_va) >= 0.05
    if tok_train_uses_nbrs and not tok_val_uses_nbrs:
        print("  -> TOKENS-ONLY learns to use neighbors on TRAIN but not VAL = OVERFIT (1222 blocks).")
        print("     The mechanism works; neighbors DO carry signal. Scale the data (more prompts)")
        print("     and/or init token embeddings from the frozen model, then the val curve should")
        print(f"     move. TRAIN flip_recovery@m0.25={tt_tr[0.25][1]:.0%} is the optimistic ceiling.")
    elif not tok_train_uses_nbrs:
        print("  -> TOKENS-ONLY doesn't use neighbors even on TRAIN (flat train acc across reveal).")
        print("     A from-scratch transformer can't learn the neighbor function from ~39k tokens.")
        print("     This probe is too small to answer the question -> need much more data, OR pivot")
        print("     to the parameter-free refresh probe / the real retrain.")
    else:
        print("  -> TOKENS-ONLY uses neighbors on BOTH train and val. Read its VAL flip_recovery@m0.25")
        print(f"     = {tt_va[0.25][1]:.0%} vs anchor-only ~10%: that's the real recoverability signal.")
    print(f"  (with-anchor TRAIN acc lift {lift(at_tr):+.1%}: near-zero => the anchor is a sufficient")
    print("   per-position shortcut, so that model never needs neighbors — expected, not a bug.)")
    print("=" * 78)


if __name__ == "__main__":
    main()
