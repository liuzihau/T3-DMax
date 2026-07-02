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
    """Merged heavy-fine-tune step (see module docstring). micro_batch carries the dual stream (built by the
    trainer): input_ids=[noisy|clean] [B,2L], attention_mask block-diffusion [B,1,2L,2L], position_ids [B,2L],
    noisy_input_ids [B,L]. Returns loss/n_micro_batches (and a metrics dict if return_metrics). B=1 assumed."""
    core = model.module if hasattr(model, "module") else model
    cfg = core.config
    bs = args.train.block_size

    # ---- per-example knobs ----
    thr_lo = _cfg(args, "heavy_commit_threshold_low", 0.75)
    thr_hi = _cfg(args, "heavy_commit_threshold_high", 0.9)
    heavy_thr = random.uniform(thr_lo, thr_hi)
    draft_k_choices = [int(k) for k in str(_cfg(args, "draft_top_k_choices", "1,2,3")).split(",")]
    draft_k = random.choice(draft_k_choices)
    heavy_top_k = int(_cfg(args, "heavy_soft_top_k", 1))
    heavy_tau = float(_cfg(args, "heavy_soft_tau", 1.0))
    draft_tau = float(_cfg(args, "draft_soft_tau", 1.0))
    alpha = float(_cfg(args, "loss_b_weight", 1.0))          # mask-denoise route
    beta = float(_cfg(args, "loss_a_weight", 1.0))           # draft-correct route

    full = micro_batch["input_ids"]                          # [B,2L] = [noisy | clean]
    attn = micro_batch["attention_mask"]
    pos = micro_batch["position_ids"]
    L = micro_batch["noisy_input_ids"].shape[1]
    noisy_ids, clean_ids = full[:, :L], full[:, L:]          # clean_ids = golden
    masked = (noisy_ids == mask_id)                          # [B,L] region to denoise
    w = masked.float()
    denom = w.sum().clamp_min(1.0)
    V = cfg.vocab_size
    embed = core.draft.frozen_embed

    def _ce_on_masked(noisy_logits):
        ce = F.cross_entropy(noisy_logits.reshape(-1, noisy_logits.shape[-1]).float(),
                             clean_ids.reshape(-1), reduction="none").view_as(clean_ids)
        return (ce * w).sum() / denom

    # ================= heavy forward #1 (GRAD) : mask-denoise loss + draft ingredients =================
    hout1 = core.heavy(input_ids=full, attention_mask=attn, position_ids=pos,
                       use_cache=False, output_hidden_states=True, output_router_logits=False, return_dict=True)
    noisy_logits1 = hout1.logits[:, :L]
    loss_B = _ce_on_masked(noisy_logits1)

    # detached signals for the (frozen) drafter
    sel = torch.cat([hout1.hidden_states[i] for i in cfg.sel_layers_list], dim=-1)   # [B,2L,mD]
    noisy_h_sel = sel[:, :L].detach()
    clean_h_sel = sel[:, L:].detach()
    noisy_h_last = hout1.hidden_states[-1][:, :L].detach()
    noisy_logits1_d = noisy_logits1.detach()

    # ---- heavy commit (no_grad) : the heavy's confident masked positions ----
    post_commit, remaining = heavy_commit(noisy_logits1_d, noisy_ids, mask_id, bs, heavy_thr)
    heavy_committed = masked & (~remaining)                  # [B,L]

    # ---- draft forward (no_grad) : predict the still-masked region ----
    with torch.no_grad():
        dout = core.draft(
            input_ids=post_commit, heavy_logits=noisy_logits1_d,
            h_sel_denoise=noisy_h_sel, h_last_denoise=noisy_h_last, h_sel_prefix=clean_h_sel,
            attention_mask=derive_drafter_mask(attn, L), position_ids=pos, denoise_mask=None, tau=None,
        )
    draft_logits = dout["logits"]                            # [B,L,V] over the noisy positions

    # ================= build the noisy soft-embeds for forward #2 (detached) =================
    noisy_embeds = embed(noisy_ids).detach().clone()         # golden@revealed, embed(MASK)@masked
    if bool(heavy_committed.any()):
        noisy_embeds[heavy_committed] = soft_embed(noisy_logits1_d[heavy_committed], embed, mask_id,
                                                   heavy_tau, heavy_top_k)
    if bool(remaining.any()):
        noisy_embeds[remaining] = soft_embed(draft_logits[remaining], embed, mask_id, draft_tau, draft_k)
    clean_embeds = embed(clean_ids).detach()
    full_embeds = torch.cat([noisy_embeds, clean_embeds], dim=1)   # [B,2L,D]

    # ================= heavy forward #2 (GRAD) : verify heavy + correct draft =================
    hout2 = core.heavy(inputs_embeds=full_embeds, attention_mask=attn, position_ids=pos,
                       use_cache=False, output_router_logits=False, return_dict=True)
    noisy_logits2 = hout2.logits[:, :L]
    loss_A = _ce_on_masked(noisy_logits2)

    loss = (alpha * loss_B + beta * loss_A) / n_micro_batches
    if not return_metrics:
        return loss
    metrics = {
        "loss_B": float(loss_B.detach()), "loss_A": float(loss_A.detach()),
        "heavy_thr": heavy_thr, "draft_k": draft_k,
        "n_masked": int(masked.sum()), "n_heavy_commit": int(heavy_committed.sum()),
        "n_draft": int(remaining.sum()),
        "acc_heavy1": float(((noisy_logits1_d.argmax(-1) == clean_ids) & masked).float().sum() / denom),
        "acc_corr2": float(((noisy_logits2.argmax(-1) == clean_ids) & masked).float().sum() / denom),
    }
    return loss, metrics
