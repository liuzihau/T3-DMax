# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
"""Core step for fine-tuning the HEAVY (top-N layers) so it is robust to the drafter's top-k soft-embeds and
learns to CORRECT them toward gold -- "OPUT with draft-contaminated rollout". Merged 2-route step (one data
pass gives both losses), on the dual-stream [noisy|clean] with the block-diffusion mask UNCHANGED:

  reveal (left-to-right, ~25%) -> [golden-revealed | MASK] in the noisy half
  heavy fwd #1 (GRAD) on [noisy|clean]:
      loss_B = CE(noisy logits, gold) on masked positions            <- Route B (mask-denoise; keeps base skill)
      detach hidden -> draft signals ; heavy_commit(th~0.75-0.9) -> commit the heavy's CONFIDENT masked positions
  draft fwd (no_grad) -> top-k(1-3) on the STILL-masked positions
  noisy inputs_embeds = [ golden(revealed) ; heavy-soft-embed(committed) ; draft-soft-embed(remaining) ]
  heavy fwd #2 (GRAD) on [noisy_embeds | clean_embeds]:
      loss_A = CE(noisy logits, gold) on the originally-masked positions   <- Route A (verify heavy + correct draft)
  loss = alpha*loss_B + beta*loss_A

Only the heavy's top layers train (the drafter + heavy bottom + embed/lm_head are frozen; set by the trainer).
Reuses heavy_commit / derive_drafter_mask from dbet_train_core.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from dbet_train_core import MASK_ID, derive_drafter_mask, heavy_commit


def soft_embed(logits_sel, embed_layer, mask_id, tau, top_k):
    """DMax soft-embed for committed positions: softmax(logits/tau) -> top-k weighted token embeds +
    residual*embed(MASK), renormalized. logits_sel [n,V] -> [n,D]. (Matches generate_dbet._soft_embed.)"""
    device = logits_sel.device
    probs = torch.softmax(logits_sel.float() / max(float(tau), 1e-6), dim=-1)
    topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)
    residual = (1.0 - topk_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    topk_emb = embed_layer(topk_idx).float()
    mask_emb = embed_layer(torch.tensor([mask_id], device=device)).float()
    s = (topk_emb * topk_probs.unsqueeze(-1)).sum(dim=1) + mask_emb * residual
    tgt = (topk_emb.norm(dim=-1) * topk_probs).sum(dim=-1, keepdim=True) + mask_emb.norm() * residual
    s = s * (tgt / (s.norm(dim=-1, keepdim=True) + 1e-6))
    return s.to(embed_layer.weight.dtype)


def _cfg(args, name, default):
    return getattr(args.train, name, default)


def dmax_dbet_train_step(model, micro_batch, n_micro_batches, args, mask_id=MASK_ID, return_metrics=False):
    """Merged heavy-fine-tune step. Runs the whole 2-forward+commit+draft step THROUGH `model(...)` (the FSDP
    root forward) via `dmax_ft_kwargs`, so the sharded root params (embed/lm_head) get gathered -- direct
    submodule calls (core.heavy/core.draft) fail under FSDP2 (DTensor). micro_batch: dual stream input_ids
    [B,2L], attention_mask [B,1,2L,2L], position_ids [B,2L], noisy_input_ids [B,L]. B=1 assumed."""
    # ---- per-example knobs (sampled here, passed into the model step) ----
    thr_lo = _cfg(args, "heavy_commit_threshold_low", 0.75)
    thr_hi = _cfg(args, "heavy_commit_threshold_high", 0.9)
    heavy_thr = random.uniform(thr_lo, thr_hi)
    draft_k_choices = [int(k) for k in str(_cfg(args, "draft_top_k_choices", "1,2,3")).split(",")]
    draft_k = random.choice(draft_k_choices)
    alpha = float(_cfg(args, "loss_b_weight", 1.0))          # mask-denoise route
    beta = float(_cfg(args, "loss_a_weight", 1.0))           # draft-correct route

    loss_B, loss_A, metrics = model(dmax_ft_kwargs=dict(
        full=micro_batch["input_ids"], attention_mask=micro_batch["attention_mask"],
        position_ids=micro_batch["position_ids"], noisy_len=micro_batch["noisy_input_ids"].shape[1],
        mask_id=mask_id, block_size=args.train.block_size, heavy_thr=heavy_thr, draft_k=draft_k,
        heavy_top_k=int(_cfg(args, "heavy_soft_top_k", 1)),
        heavy_tau=float(_cfg(args, "heavy_soft_tau", 1.0)),
        draft_tau=float(_cfg(args, "draft_soft_tau", 1.0)),
    ))
    loss = (alpha * loss_B + beta * loss_A) / n_micro_batches
    return (loss, metrics) if return_metrics else loss
