"""Block 2a — tiny OVERFIT smoke for the anchor-free top-K talk training.

Confirms the TRAINING MECHANICS on the real models before committing GPU hours to a
full run: freeze the talk to the merged-representative layers only, then run an
optimizer on ONE fixed batch for N steps and check the loss DROPS toward ~0 (the talk
can memorize the batch → backward + optimizer + the merged-only freeze all work, and
the merged layers have enough capacity to move the loss). This is a wiring check, not
real learning (real learning needs the data pipeline = the Block-2 trainer).

Default trainable scope = the 2 merged layers (stack positions 6,8) per the
keep=0-5,12,19 plan — i.e. exactly the "merged layers only" run config.

Run (single GPU):
  python -m tasks.t3d_topk_talk_train_smoke \
      --think_path ../DMax-Math-16B-moe-merge --talk_path ../merged_10L \
      --mask_id 156895 --block_len 32 --steps 200 --lr 1e-4 --train_layers 6,8 --flag topk
"""

from __future__ import annotations

import argparse

import torch

from tasks.t3d_topk_talk import (
    load_causal_lm, set_talk_trainable, topk_talk_train_step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--think_path", required=True)
    ap.add_argument("--talk_path", required=True)
    ap.add_argument("--mask_id", type=int, default=156895)
    ap.add_argument("--block_len", type=int, default=32)
    ap.add_argument("--reveal_frac", type=float, default=0.25)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train_layers", default="6,8", help="talk stack indices to train (merged reps)")
    ap.add_argument("--flag", choices=["topk", "mask"], default="topk", help="Path B (topk) or A (mask)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(0)
    dtype = torch.bfloat16

    print("[train-smoke] loading think + talk…")
    think = load_causal_lm(args.think_path, args.device, dtype)
    talk = load_causal_lm(args.talk_path, args.device, dtype)
    for p in think.parameters():
        p.requires_grad_(False)
    emb = think.get_input_embeddings()

    train_idx = [int(x) for x in args.train_layers.split(",") if x != ""]
    n_train = set_talk_trainable(talk, train_idx, freeze_embed_head=True)
    print(f"[train-smoke] trainable layers {train_idx} → {n_train:,} params "
          f"({n_train/1e9:.2f}B; merged-only)")

    # ---- ONE fixed batch (memorization target) ----
    B, L = 2, args.block_len
    V = emb.weight.shape[0]
    labels = torch.randint(0, V - 1, (B, L), device=args.device)
    noisy = torch.full((B, L), args.mask_id, device=args.device, dtype=torch.long)
    n_reveal = max(1, int(args.reveal_frac * L))
    rev = torch.argsort(torch.rand(B, L, device=args.device), -1)[:, :n_reveal]
    revealed = torch.zeros(B, L, dtype=torch.bool, device=args.device)
    revealed.scatter_(1, rev, True)
    noisy[revealed] = labels[revealed]
    labels_ce = labels.clone()
    labels_ce[revealed] = -100
    attn = torch.zeros(B, 1, L, L, device=args.device, dtype=dtype)     # bidirectional (single block)
    pos = torch.arange(L, device=args.device).unsqueeze(0).expand(B, -1)

    opt = torch.optim.AdamW([p for p in talk.parameters() if p.requires_grad], lr=args.lr)
    talk.train()
    flag = (args.flag == "topk")
    losses = []
    for step in range(args.steps):
        loss = topk_talk_train_step(think, talk, emb, args.mask_id, noisy, labels_ce,
                                    attn, pos, flag=flag, top_k=args.top_k)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(f"[train-smoke] step {step:4d}  loss {loss.item():.4f}")

    drop = losses[0] - losses[-1]
    print(f"[train-smoke] loss {losses[0]:.3f} → {losses[-1]:.3f}  (Δ={drop:.3f})")
    assert losses[-1] < 0.5 * losses[0], "loss did not drop — training mechanics broken"
    print("[train-smoke] PASS — loss drops; merged-only training mechanics work. "
          "Ready to apply the Block-2 patch and train on real data.")


if __name__ == "__main__":
    main()
