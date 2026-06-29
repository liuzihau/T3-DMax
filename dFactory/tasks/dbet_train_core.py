# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
"""DBet core training step (pure torch — no VeOmni), so it can be unit/smoke-tested off-cluster.

One FROZEN-heavy dual-stream forward -> DMax decode_uniform commit (one pass) -> drafter forward over
[prefix+clean ; noisy] -> decayed CE + confidence BCE on the remaining-masked vs golden. Imported by
`train_dbet.py` (the VeOmni trainer) and by `smoke_dbet.py` (the off-cluster test). `args` is duck-typed:
needs `args.train.block_size`, `args.train.heavy_commit_threshold`, `args.train.conf_loss_weight`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

MASK_ID = 156895


def heavy_commit(noisy_logits, noisy_ids, mask_id, block_size, threshold):
    """DMax decode_uniform commit — ONE pass, argmax, per block, left-to-right prefix of masked positions until
    the first below-threshold (no fallback: training needs no guaranteed progress).
    noisy_logits [B,L,V], noisy_ids [B,L] -> (post_commit_ids [B,L], remaining_mask [B,L] bool)."""
    B, L, _ = noisy_logits.shape
    nb = L // block_size
    probs = torch.softmax(noisy_logits.float(), dim=-1)
    argmax = probs.argmax(dim=-1)                                       # [B,L]
    conf = probs.gather(-1, argmax.unsqueeze(-1)).squeeze(-1)           # [B,L]
    mask = noisy_ids == mask_id
    is_low = (mask & (conf < threshold)).view(B, nb, block_size)
    has_failed = torch.cumsum(is_low.long(), dim=-1) > 0               # any low-conf masked at/before, per block
    commit = (mask.view(B, nb, block_size) & (~has_failed)).view(B, L)
    post = torch.where(commit, argmax, noisy_ids)
    remaining = mask & (~commit)
    return post, remaining


def derive_drafter_mask(dual_mask, L):
    """Drafter mask = noisy-query rows of the dual-stream prototype, columns reordered to [clean ; noisy]
    (matching DbetAttention keys = [prefix_kv(=clean) ; canvas(=noisy)]). dual_mask [B,1,2L,2L] over
    [noisy(0:L) | clean(L:2L)] -> [B,1,L,2L]. Noisy block i attends clean blocks < i (M_OBC) + own noisy
    block (M_BD); clean includes the prompt (early blocks) so the prompt is attended by all."""
    noisy_rows = dual_mask[:, :, :L, :]
    return torch.cat([noisy_rows[:, :, :, L:2 * L], noisy_rows[:, :, :, :L]], dim=-1)


def decay_weights(remaining_mask, block_size):
    """Per-block left-to-right decay over remaining positions: w[k] = max(0.9^k, 0.1), k=0 at the first
    remaining position in the block, 0 outside remaining. remaining_mask [B,L] -> w [B,L]."""
    B, L = remaining_mask.shape
    nb = L // block_size
    rem = remaining_mask.view(B, nb, block_size)
    k = (torch.cumsum(rem.long(), dim=-1) - 1).clamp(min=0)
    w = torch.clamp(0.9 ** k.float(), min=0.1) * rem.float()
    return w.view(B, L)


def dbet_forward(core, micro_batch, args, mask_id=MASK_ID, return_post_commit=False):
    """Shared FROZEN-heavy -> commit -> drafter forward (no loss). Used by BOTH `dbet_train_step` (with grad on
    the drafter) and the eval pass (wrapped in no_grad) so the two can never drift apart.
    `core` is the UNWRAPPED model; micro_batch carries the dual stream (input_ids=[noisy|clean] [B,2L],
    attention_mask=[B,1,2L,2L] block prototype, position_ids=[B,2L], noisy_input_ids=[B,L]).
    Returns: logits [B,L,V], conf [B,L] (or None), remaining [B,L] bool, clean_ids [B,L] (golden).
    If return_post_commit: also returns post_commit [B,L] (heavy pass-1 committed ids) as a 5th element
    (the eval uses it to run a heavy SECOND pass)."""
    cfg = core.config
    bs, thr = args.train.block_size, args.train.heavy_commit_threshold

    full = micro_batch["input_ids"]                                    # [B, 2L] = [noisy | clean]
    attn = micro_batch["attention_mask"]
    pos = micro_batch["position_ids"]
    L = micro_batch["noisy_input_ids"].shape[1]
    noisy_ids, clean_ids = full[:, :L], full[:, L:]                    # clean_ids = golden answer

    # 1) frozen heavy dual-stream forward (its TRAINED layout -> valid hidden); harvest both halves
    with torch.no_grad():
        hout = core.heavy(input_ids=full, attention_mask=attn, position_ids=pos,
                          use_cache=False, output_hidden_states=True, output_router_logits=False, return_dict=True)
    h_sel = torch.cat([hout.hidden_states[i] for i in cfg.sel_layers_list], dim=-1)   # [B,2L,m*D]
    noisy_h_sel, clean_h_sel = h_sel[:, :L], h_sel[:, L:]
    noisy_h_last = hout.hidden_states[-1][:, :L]
    noisy_logits = hout.logits[:, :L]

    # 2) heavy one-pass decode_uniform commit on the noisy logits -> committed / remaining
    post_commit, remaining = heavy_commit(noisy_logits, noisy_ids, mask_id, bs, thr)

    # 3) drafter forward: clean(+prompt) hidden -> prefix KV; noisy -> canvas; mask = [clean ; noisy]
    out = core.draft(
        input_ids=post_commit, heavy_logits=noisy_logits,
        h_sel_denoise=noisy_h_sel, h_last_denoise=noisy_h_last, h_sel_prefix=clean_h_sel,
        attention_mask=derive_drafter_mask(attn, L), position_ids=pos, denoise_mask=None, tau=None,
    )
    if return_post_commit:
        return out["logits"], out["conf"], remaining, clean_ids, post_commit
    return out["logits"], out["conf"], remaining, clean_ids


def dbet_train_step(model, micro_batch, n_micro_batches, args, mask_id=MASK_ID, return_metrics=False):
    """DBet core step (requires the dual stream already in micro_batch: input_ids=[noisy|clean] [B,2L],
    attention_mask=[B,1,2L,2L] block-diffusion prototype, position_ids=[B,2L], noisy_input_ids=[B,L]).
    Returns loss/n_micro_batches (and a metrics dict if return_metrics)."""
    core = model.module if hasattr(model, "module") else model         # unwrap FSDP1 if present
    bs = args.train.block_size

    logits, conf, remaining, clean_ids = dbet_forward(core, micro_batch, args, mask_id)

    # decayed CE + confidence BCE on the remaining-masked positions vs golden
    w = decay_weights(remaining, bs)
    denom = w.sum().clamp_min(1.0)
    # CE in fp32: bf16 logsumexp can overflow for large drafter logits (NaN). logits.float() is the stable path.
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), clean_ids.reshape(-1),
                         reduction="none").view_as(clean_ids)
    tok_loss = (ce * w).sum() / denom
    loss = tok_loss
    metrics = {"tok": float(tok_loss.detach()), "n_remaining": int(remaining.sum())}
    if conf is not None:
        accept = (logits.argmax(-1) == clean_ids).float()             # label 1 iff drafter argmax == golden
        c = conf.float().clamp(1e-5, 1 - 1e-5)                         # fp32 for a stable log
        bce = -(accept * c.log() + (1 - accept) * (1 - c).log())
        conf_loss = (bce * w).sum() / denom
        loss = loss + args.train.conf_loss_weight * conf_loss
        metrics["conf"] = float(conf_loss.detach())
        metrics["acc"] = float((accept * w).sum() / denom)            # decayed drafter accuracy on remaining
    loss = loss / n_micro_batches
    return (loss, metrics) if return_metrics else loss
