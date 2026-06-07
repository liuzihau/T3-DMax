"""Block 1 — anchor-free top-K talk: single-GPU FORWARD smoke test.

Verifies the new forward path end-to-end on the REAL two models, isolated from the
VeOmni/FSDP training machinery (that's Block 2). With the merge_layers init the talk
is just a 10-layer LLaDA2-Moe; think is the full 20-layer. They share the input
embedding + lm_head (frozen). The talk runs ANCHOR-FREE on inputs_embeds where the
still-masked positions carry think's top-K candidate blend.

What it checks:
  1. both models load; talk has fewer layers than think;
  2. think (frozen, no grad) -> top-K candidates;
  3. build_talk_inputs_embeds injects the top-K at masked positions;
  4. talk forward -> logits -> CE on the masked (predict) positions;
  5. loss is finite and backward populates grads ONLY on the talk layers
     (think / embedding / lm_head get no grad).

Run (single GPU):
  python -m tasks.t3d_topk_talk_smoke \
      --think_path /path/to/DMax-Math-16B-moe-merge \
      --talk_path  /path/to/merged_10L \
      --mask_id 156895 --block_len 32 --reveal_frac 0.25

If the talk checkpoint isn't ready yet, build it first:
  python -m probe_runner.eval_layer_subset ... is NOT this; use:
  python -m probe_runner.merge_layers --model_path <think> --keep 0-5,12,19 \
      --n_merged_per_block 1 --dry_run --save_hf /path/to/merged_10L
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from tasks.t3d_topk_talk import (
    build_talk_inputs_embeds, predict_loss, topk_talk_train_step, load_causal_lm)


def _decoder_layers(m):
    base = getattr(m, "model", m)
    return [x for x in base.modules() if type(x).__name__ == "LLaDA2MoeDecoderLayer"]


def _block_bidirectional_mask(batch, seq_len, device, dtype):
    """Smoke-only: full bidirectional attention over the single block, per-sample
    [B,1,L,L] additive mask (0 = attend). The real training uses the block-causal
    mask (prompt visible, causal across blocks)."""
    return torch.zeros(batch, 1, seq_len, seq_len, device=device, dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--think_path", required=True)
    ap.add_argument("--talk_path", required=True, help="merge_layers --save_hf 10-layer ckpt")
    ap.add_argument("--mask_id", type=int, default=156895)
    ap.add_argument("--block_len", type=int, default=32)
    ap.add_argument("--reveal_frac", type=float, default=0.25)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(0)
    dtype = torch.bfloat16

    print("[smoke] loading think (full) and talk (10-layer merge)…")
    think = load_causal_lm(args.think_path, args.device, dtype)
    talk = load_causal_lm(args.talk_path, args.device, dtype)
    n_think, n_talk = len(_decoder_layers(think)), len(_decoder_layers(talk))
    print(f"[smoke] think layers={n_think}  talk layers={n_talk}")
    assert n_talk < n_think, "talk should be the compressed model"

    # SHARE embedding + lm_head from think (frozen); train only the talk's layers.
    emb = think.get_input_embeddings()
    head = think.get_output_embeddings() if think.get_output_embeddings() is not None else think.lm_head
    for p in think.parameters():
        p.requires_grad_(False)
    for p in talk.get_input_embeddings().parameters():
        p.requires_grad_(False)
    talk_head = talk.get_output_embeddings() if talk.get_output_embeddings() is not None else talk.lm_head
    for p in talk_head.parameters():
        p.requires_grad_(False)
    n_train = sum(p.numel() for p in talk.parameters() if p.requires_grad)
    print(f"[smoke] trainable (talk layers): {n_train:,}")

    # ---- a synthetic block: labels = random tokens; reveal reveal_frac as context ----
    B, L = 2, args.block_len
    V = head.weight.shape[0]
    labels = torch.randint(0, V - 1, (B, L), device=args.device)
    noisy = torch.full((B, L), args.mask_id, device=args.device, dtype=torch.long)
    n_reveal = max(1, int(args.reveal_frac * L))
    rev_idx = torch.argsort(torch.rand(B, L, device=args.device), dim=-1)[:, :n_reveal]
    revealed = torch.zeros(B, L, dtype=torch.bool, device=args.device)
    revealed.scatter_(1, rev_idx, True)
    noisy[revealed] = labels[revealed]                          # committed context
    still_masked = (noisy == args.mask_id)
    labels_for_ce = labels.clone()
    labels_for_ce[revealed] = -100                              # don't score revealed context

    attn = _block_bidirectional_mask(B, L, args.device, dtype)
    pos = torch.arange(L, device=args.device).unsqueeze(0).expand(B, -1)

    # Sanity: build_talk_inputs_embeds injects think's top-K at masked positions only.
    with torch.no_grad():
        think_logits = think(inputs_embeds=emb(noisy), attention_mask=attn,
                             position_ids=pos, use_cache=False, return_dict=True).logits
    print(f"[smoke] think_logits {tuple(think_logits.shape)}")
    talk_embeds = build_talk_inputs_embeds(noisy, think_logits, emb, args.mask_id,
                                           mode="topk_soft", top_k=args.top_k, keep_mask_residual=False)
    plain = emb(noisy)
    assert not torch.allclose(talk_embeds[still_masked], plain[still_masked])
    assert torch.allclose(talk_embeds[~still_masked], plain[~still_masked])
    print(f"[smoke] top-K injected at {int(still_masked.sum())} masked / "
          f"{int((~still_masked).sum())} committed positions")

    # THE training step (the exact function Block 2's loop calls). Test both paths.
    talk.train()
    loss_false = topk_talk_train_step(think, talk, emb, args.mask_id, noisy, labels_for_ce,
                                      attn, pos, flag=False, top_k=args.top_k)   # Path A ([MASK])
    loss = topk_talk_train_step(think, talk, emb, args.mask_id, noisy, labels_for_ce,
                                attn, pos, flag=True, top_k=args.top_k)          # Path B (top-K)
    print(f"[smoke] train-step loss: flag=False(mask)={loss_false.item():.4f}  "
          f"flag=True(top-K)={loss.item():.4f}  (finite={torch.isfinite(loss).item()})")
    assert torch.isfinite(loss) and torch.isfinite(loss_false)

    # backward (on the Path-B loss): grads only on talk layers; think/emb/head get none
    loss.backward()
    talk_layer_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in talk.parameters() if p.requires_grad)
    think_grad = any(p.grad is not None for p in think.parameters())
    emb_grad = any(p.grad is not None for p in emb.parameters())
    assert talk_layer_grad, "talk layers should receive gradient"
    assert not think_grad and not emb_grad, "think / embedding must stay frozen (no grad)"
    print("[smoke] grad check OK: talk layers grad ✓, think/embed frozen ✓")
    print("[smoke] PASS — anchor-free top-K forward works end-to-end. Ready for Block 2 "
          "(wire into the VeOmni training loop).")


if __name__ == "__main__":
    main()
