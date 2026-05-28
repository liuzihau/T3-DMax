# Copyright 2026 University of Sydney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This file is derived from `dFactory/tasks/train_llada2_bd_oput.py` in DMax
# (https://github.com/czg1225/DMax), Copyright 2026 National University of Singapore,
# Apache-2.0.
#
# Modifications by University of Sydney for T3-D milestone 1:
#   - Register the `think_talk_llada2` model instead of `llada2_moe`.
#   - Add `T3TrainingArguments` (rollout target, train iterations, mask token id).
#   - Replace the no-grad rollout block (originally L448-L472) with a talk-only rollout
#     that reuses the think anchor from the masked input. See brief sec 8.3.
#   - Add a one-shot sanity log of the anchor-leak property at step 1.
#
# Every modification vs. the original DMax file is annotated with a `# T3-D ...:` comment.

import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Tuple, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from tqdm import trange

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    build_dataloader,
    build_iterative_dataset,
    build_mapping_dataset,
)
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.device import (
    get_device_type,
    get_nccl_backend,
    get_torch_device,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce
from veomni.models.registry import ModelRegistry

# T3-D MODIFIED: register the Think-Then-Talk model module (was "models.llada2_moe").
ModelRegistry.register_modeling_path("models.think_talk_llada2")

from dataset.data_transform import process_mdm_tokenized_example, process_mdm_sft_example
from dataset import build_local_dataset
import random


logger = helper.create_logger(__name__)


@dataclass
class LLaDA2ModelArguments(ModelArguments):
    """Unchanged from DMax."""
    attn_implementation: Optional[Literal["eager", "sdpa", "flex_attention"]] = field(
        default="sdpa",
        metadata={"help": "Attention implementation to use."},
    )


@dataclass
class LLaDA2DataArguments(DataArguments):
    """Unchanged from DMax."""
    data_type: Literal["conversation", "tokenid"] = field(default="conversation")
    datasets_type: Literal["mapping", "local"] = field(default="mapping")
    text_keys: str = field(default="messages")
    noise_range_low: float = field(default=0.3)
    noise_range_high: float = field(default=0.8)

    def __post_init__(self):
        super().__post_init__()
        if self.noise_range_low > self.noise_range_high:
            raise ValueError(
                f"noise_range_low ({self.noise_range_low}) cannot be greater than "
                f"noise_range_high ({self.noise_range_high})."
            )
        if not (0.0 <= self.noise_range_low <= 1.0):
            raise ValueError(f"noise_range_low must be in [0,1], got {self.noise_range_low}")
        if not (0.0 <= self.noise_range_high <= 1.0):
            raise ValueError(f"noise_range_high must be in [0,1], got {self.noise_range_high}")


@dataclass
class LLaDA2TrainingArguments(TrainingArguments):
    """Unchanged from DMax."""
    beta1: float = field(default=0.9)
    beta2: float = field(default=0.999)
    block_diffusion_mode: bool = field(default=False)
    block_size: int = field(default=32)
    same_token_labels: bool = field(default=False)


# T3-D ADDED: training-side knobs that only T3-D understands.
@dataclass
class T3TrainingArguments(LLaDA2TrainingArguments):
    t3_rollout_mode: Literal["dmax_oput", "none"] = field(
        default="dmax_oput",
        metadata={"help": "OPUT mode. 'dmax_oput' replaces masked positions with talk argmax "
                          "on flag=True samples. 'none' disables the OPUT rollout entirely."},
    )
    t3_rollout_target: Literal["talk_only", "think_and_talk"] = field(
        default="talk_only",
        metadata={"help": "What to recompute on flag=True. 'talk_only' (default, brief sec 8.3) "
                          "reuses the masked-input think anchor; 'think_and_talk' recomputes both "
                          "(strict DMax compute pattern, kept as ablation)."},
    )
    t3_rollout_replace: Literal["all_masked", "confidence"] = field(
        default="all_masked",
        metadata={"help": "Which masked positions to replace with the rollout's argmax. "
                          "Milestone-1 default 'all_masked' matches DMax."},
    )
    t3_train_iterations: int = field(
        default=1,
        metadata={"help": "Number of talk-side denoising iterations per grad step. "
                          "Set >1 for multi-iter training (A4): each step runs N talk "
                          "forwards through with grad, with reveal between iterations. "
                          "Loss is the uniform average of per-iter CE."},
    )
    t3_reveal_threshold: float = field(
        default=0.5,
        metadata={"help": "Softmax-peak threshold for DMax-uniform reveal (mask path "
                          "between training iterations). DMax's released code uses 0.5."},
    )
    mask_token_id: int = field(
        default=156895,
        metadata={"help": "LLaDA-2.0-mini's [MASK] token id."},
    )
    # T3-D ADDED: differential LR ratio for LM head (Strategy C). 1.0 -> same LR as `lr`.
    # Typical value 0.02-0.05 puts LM head at fine-tune scale while talk learns at scratch
    # scale. Only takes effect when train_lm_head=true (otherwise LM head is frozen).
    lr_lm_head_ratio: float = field(
        default=1.0,
        metadata={"help": "Multiplier applied to `lr` for the lm_head param group. "
                          "1.0 = single-group optimizer (no split). 0.02-0.05 = differential "
                          "LR (Strategy C) keeping LM head near its DMax fine-tune LR."},
    )
    # T3-D ADDED: rollout-flag ratio ramp. When low != high, each micro_batch's flag is
    # resampled as Bernoulli(threshold) where threshold ramps linearly across training.
    # Default 0.5/0.5 preserves the dataset's 50/50 flag distribution untouched.
    #
    # Setting low=0.25, high=0.75 trains mostly on the easier mask path early (75% of
    # batches), then shifts to the harder rollout path late (75% of batches). This is a
    # curriculum: early gradients come from the standard masked distribution that LLaDA
    # was originally trained on; late gradients come from the OPUT distribution that
    # inference will use. Complements the noise-range ramp.
    t3_rollout_ratio_low: float = field(
        default=0.5,
        metadata={"help": "Probability that a micro_batch follows the rollout path at "
                          "training start. Each step: flag := (rand() < threshold)."},
    )
    t3_rollout_ratio_high: float = field(
        default=0.5,
        metadata={"help": "Probability that a micro_batch follows the rollout path at "
                          "training end. Linearly ramped from t3_rollout_ratio_low."},
    )
    # T3-D ADDED: inline validation knobs. Validation runs the deterministic tail of the
    # training data through model.forward (single iter, no rollout) at fixed sigmas, and
    # logs CE split into overall / mask-region / clean-region per sigma. Useful for
    # tracking real progress past the noisy training-loss curve (which mixes per-step
    # sigma noise from the ramp).
    t3_val_every: int = field(
        default=0,
        metadata={"help": "Run validation every N global steps (in addition to step 0 "
                          "baseline). 0 disables inline validation."},
    )
    t3_val_tail: int = field(
        default=50,
        metadata={"help": "Number of samples (from the tail of the seed-shuffled train "
                          "data) to use as the inline-validation set. Same shuffle "
                          "convention as tasks/eval_ce_val.py."},
    )
    t3_val_sigmas: str = field(
        default="0.5,0.75",
        metadata={"help": "Comma-separated sigma values to evaluate at."},
    )


@dataclass
class Arguments:
    model: "LLaDA2ModelArguments" = field(default_factory=LLaDA2ModelArguments)
    data: "LLaDA2DataArguments" = field(default_factory=LLaDA2DataArguments)
    # T3-D MODIFIED: use T3-extended training args.
    train: "T3TrainingArguments" = field(default_factory=T3TrainingArguments)


def block_diffusion_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """Unchanged from DMax. Builds the doubled-sequence block-diffusion attention mask
    composed of three sub-masks: M_BD (block diagonal, intra-block self-attn within xt or
    x0), M_OBC (offset block causal, xt attends to earlier x0 blocks), M_BC (block causal
    within x0). The crucial property: xt positions cannot attend to x0 positions in the
    same or future blocks -- so x0 (clean labels) can never leak into the xt anchor.

    Verified by `tests/test_anchor_leak.py`.
    """
    x0_flag_q = (q_idx >= n)
    x0_flag_kv = (kv_idx >= n)

    block_q = torch.where(x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size)

    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    return block_diagonal | offset_block_causal | block_causal


# T3-D ADDED: 3L attention mask for the concat_segment anchor-injection mode.
#
# Sequence layout: [noisy(0..n-1), anchor(n..2n-1), clean(2n..3n-1)]. The anchor stream
# is *attentionally identical* to noisy -- same block constraints, same access to prior
# clean blocks. Implementation: remap each position to its "effective" position within
# its conceptual region (noisy_or_anchor vs clean), then apply the same M_BD / M_OBC /
# M_BC submasks as the 2L case.
#
# Verified mask cells (q region -> k region):
#   noisy  -> noisy  : M_BD                 (same block)
#   noisy  -> anchor : M_BD                 (same block)
#   noisy  -> clean  : M_OBC                (prior blocks only)
#   anchor -> noisy  : M_BD                 (same block)
#   anchor -> anchor : M_BD                 (same block)
#   anchor -> clean  : M_OBC                (prior blocks only)
#   clean  -> noisy  : blocked
#   clean  -> anchor : blocked
#   clean  -> clean  : M_BC                 (block-causal among clean)
#
# Crucially, anchor queries cannot see clean keys in their own or future blocks ->
# anchor[i] never leaks label info for block of i.
def block_diffusion_mask_3L(q_idx, kv_idx, block_size, n):
    # "Is this position in the clean region?" -- clean starts at 2n.
    x0_flag_q  = (q_idx  >= 2 * n)
    x0_flag_kv = (kv_idx >= 2 * n)

    # Effective intra-region position: noisy keeps its index, anchor maps to its
    # noisy-equivalent (idx - n), clean maps to (idx - 2n). Anchor and noisy then
    # share the same block index for the block-membership check below.
    eff_q  = torch.where(
        q_idx  < n, q_idx,
        torch.where(q_idx  < 2 * n, q_idx  - n, q_idx  - 2 * n),
    )
    eff_kv = torch.where(
        kv_idx < n, kv_idx,
        torch.where(kv_idx < 2 * n, kv_idx - n, kv_idx - 2 * n),
    )
    block_q  = eff_q  // block_size
    block_kv = eff_kv // block_size

    block_diagonal      = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    offset_block_causal = (block_q >  block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)
    block_causal        = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)
    return block_diagonal | offset_block_causal | block_causal


# T3-D ADDED: masks for the hybrid_xattn talk pathway.
#
# In hybrid_xattn mode, talk's sequence is just the L noisy tokens (no clean, no anchor
# inline). Two attention patterns:
#
# 1. Talk self-attn  (over L noisy):
#       noisy[i] in block b can self-attend to noisy[j] iff j is in the same block.
#       This is the M_BD (block-diagonal) restriction from the 2L mask, applied to L.
#
# 2. Talk cross-attn (Q from noisy[L], KV from anchor[:, :2L, :]):
#       KV columns 0..L-1 are anchor at noisy positions: always visible (already obey the
#           block-diff constraints from think's own forward).
#       KV columns L..2L-1 are anchor at clean positions: M_OBC restriction --
#           noisy_q in block b can see anchor_clean in block c iff c < b (strict, prior).
#
# Both functions return bool tensors (True = allowed). Caller converts to additive masks.
def talk_self_attn_mask_L(q_idx, kv_idx, block_size):
    block_q  = q_idx  // block_size
    block_kv = kv_idx // block_size
    return block_q == block_kv


def talk_cross_attn_mask(q_idx, kv_idx, block_size, n):
    """q_idx: L noisy positions. kv_idx: 2L anchor positions [noisy_half(0..n-1), clean_half(n..2n-1)]."""
    kv_is_clean = (kv_idx >= n)
    kv_eff_pos = torch.where(kv_is_clean, kv_idx - n, kv_idx)
    q_block  = q_idx // block_size
    kv_block = kv_eff_pos // block_size

    noisy_kv_always_ok = ~kv_is_clean
    clean_kv_prior_block = kv_is_clean & (q_block > kv_block)
    return noisy_kv_always_ok | clean_kv_prior_block


# T3-D ADDED: between-iteration reveal helpers (A4 multi-iter training).
# Both helpers take talk's noisy-half logits and the current noisy input, and return
# an updated noisy input where some [MASK] positions have been replaced with the model's
# argmax. The reveal strategy differs per OPUT flag.
#
# `reveal_dmax_uniform` (mask path, flag=False):
#   For each block of `block_size` positions, scan left-to-right; commit positions
#   whose softmax peak > threshold; stop at the first that fails. If no position in
#   a block qualifies and that block still has [MASK]s, commit the leftmost masked
#   position (guaranteed-progress fallback). Mirrors DMax inference's reveal rule.
#
# `reveal_full_argmax` (rollout path, flag=True):
#   Replace every currently-masked position with the model's argmax. No threshold,
#   no left-to-right gating. Harder training distribution: simulates "what if my
#   bulk prediction was just argmaxed without confidence gating?".
def reveal_dmax_uniform(
    logits: torch.Tensor,
    current_noisy: torch.Tensor,
    mask_token_id: int,
    block_size: int,
    threshold: float,
) -> torch.Tensor:
    B, L = current_noisy.shape
    probs = torch.softmax(logits.float(), dim=-1)
    max_probs, argmax_ids = probs.max(dim=-1)        # both [B, L]
    masked = (current_noisy == mask_token_id)         # [B, L]

    num_blocks = L // block_size
    confident_blocks = (max_probs > threshold).view(B, num_blocks, block_size).long()
    cum_conf = torch.cumprod(confident_blocks, dim=-1).bool().view(B, L)
    commit_mask = cum_conf & masked                    # [B, L]

    # Guaranteed-progress fallback per (sample, block).
    any_commit = commit_mask.view(B, num_blocks, block_size).any(dim=-1)         # [B, nb]
    masked_blocks = masked.view(B, num_blocks, block_size)                        # [B, nb, bs]
    has_mask = masked_blocks.any(dim=-1)                                          # [B, nb]
    needs_fallback = (~any_commit) & has_mask                                     # [B, nb]
    if bool(needs_fallback.any()):
        first_mask_idx = masked_blocks.long().argmax(dim=-1)                      # [B, nb]
        block_offset = (
            torch.arange(num_blocks, device=current_noisy.device) * block_size
        )                                                                          # [nb]
        abs_pos = block_offset[None] + first_mask_idx                              # [B, nb]
        batch_idx = (
            torch.arange(B, device=current_noisy.device)[:, None]
            .expand(-1, num_blocks)
        )                                                                          # [B, nb]
        flat_b = batch_idx[needs_fallback]
        flat_p = abs_pos[needs_fallback]
        fallback_mask = torch.zeros(
            B, L, dtype=torch.bool, device=current_noisy.device,
        )
        fallback_mask[flat_b, flat_p] = True
        commit_mask = commit_mask | fallback_mask

    return torch.where(commit_mask, argmax_ids, current_noisy)


def reveal_full_argmax(
    logits: torch.Tensor,
    current_noisy: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    argmax_ids = logits.argmax(dim=-1)
    masked = (current_noisy == mask_token_id)
    return torch.where(masked, argmax_ids, current_noisy)


# T3-D ADDED: inline validation. Builds a transform with fixed sigma, runs the model
# through `forward` (single iter, no rollout, no multi-iter) on the val examples, and
# returns CE split into overall / mask-region / clean-region per sigma.
#
# Mirrors `tasks/eval_ce_val.py` semantics so numbers are comparable.
@torch.no_grad()
def _run_validation(
    model,
    val_raw_examples,
    val_indices,
    sigmas,
    *,
    tokenizer,
    max_seq_len,
    block_size,
    mask_token_id,
    text_keys,
    block_diffusion_attn_mask_prototype,
    block_diffusion_attn_mask_prototype_3L,
    talk_self_attn_mask_prototype_L,
    talk_cross_attn_mask_prototype,
    device,
):
    """Inline single-forward CE validation. Returns dict of wandb-keyed metrics."""
    model.eval()

    L = max_seq_len
    # Pre-build per-batch=1 position_ids (same convention as training).
    noisy_pos = torch.arange(L, dtype=torch.long, device=device)
    clean_pos = torch.arange(L, dtype=torch.long, device=device)
    pos_2L = torch.cat([noisy_pos, clean_pos], dim=0).unsqueeze(0)
    pos_L = noisy_pos.unsqueeze(0)
    cross_pos = torch.cat([noisy_pos, clean_pos], dim=0).unsqueeze(0)

    # Reusable masks (prototypes are already shape [1,1,...] which works for batch=1).
    attn_mask_2L = block_diffusion_attn_mask_prototype.to(device, non_blocking=True)
    attn_mask_3L = (
        block_diffusion_attn_mask_prototype_3L.to(device, non_blocking=True)
        if block_diffusion_attn_mask_prototype_3L is not None else None
    )
    attn_mask_L = (
        talk_self_attn_mask_prototype_L.to(device, non_blocking=True)
        if talk_self_attn_mask_prototype_L is not None else None
    )
    cross_attn_mask = (
        talk_cross_attn_mask_prototype.to(device, non_blocking=True)
        if talk_cross_attn_mask_prototype is not None else None
    )

    metrics = {}
    for sigma in sigmas:
        transform = partial(
            process_mdm_sft_example,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            text_keys=text_keys,
            noise_range=(sigma, sigma),
            mask_token_id=mask_token_id,
            progress_state=None,
        )

        loss_sum = 0.0
        pos_count = 0
        mask_loss_sum = 0.0
        mask_count = 0
        clean_loss_sum = 0.0
        clean_count = 0
        skipped = 0

        for idx in val_indices:
            try:
                transformed = transform(val_raw_examples[idx])[0]
            except Exception:
                skipped += 1
                continue

            noisy = transformed["noisy_input_ids"]
            clean = transformed["input_ids"]
            labels = transformed["labels"]

            full_ids = torch.cat([noisy, clean], dim=0).unsqueeze(0).to(device)
            labels_dev = labels.unsqueeze(0).to(device)
            noisy_dev = noisy.unsqueeze(0).to(device)

            kwargs = {
                "input_ids": full_ids,
                "attention_mask": attn_mask_2L,
                "position_ids": pos_2L,
                "use_cache": False,
                "output_router_logits": False,
            }
            if attn_mask_3L is not None:
                kwargs["attention_mask_3L"] = attn_mask_3L
                kwargs["position_ids_3L"] = torch.cat(
                    [noisy_pos, noisy_pos, clean_pos], dim=0,
                ).unsqueeze(0)
            if attn_mask_L is not None:
                kwargs["attention_mask_L"] = attn_mask_L
                kwargs["position_ids_L"] = pos_L
                kwargs["cross_attention_mask"] = cross_attn_mask
                kwargs["cross_position_ids"] = cross_pos

            out = model(**kwargs)
            logits = out.logits
            # In hybrid_xattn talk outputs [B,L,V]; in other modes [B,2L,V] or [B,3L,V].
            # Slice to first L (the noisy half / talk's predictions over noisy positions).
            noisy_logits = logits[:, :L]

            per_pos = torch.nn.functional.cross_entropy(
                noisy_logits.view(-1, noisy_logits.shape[-1]),
                labels_dev.view(-1),
                reduction="none",
                ignore_index=-100,
            )
            valid_flat = (labels_dev != -100).view(-1)
            noisy_flat = noisy_dev.view(-1)
            mask_region = (noisy_flat == mask_token_id) & valid_flat
            clean_region = (noisy_flat != mask_token_id) & valid_flat

            loss_sum += float(per_pos.sum().item())
            pos_count += int(valid_flat.sum().item())
            mask_loss_sum += float(per_pos[mask_region].sum().item())
            mask_count += int(mask_region.sum().item())
            clean_loss_sum += float(per_pos[clean_region].sum().item())
            clean_count += int(clean_region.sum().item())

        key = f"sigma_{sigma:.2f}"
        if pos_count > 0:
            metrics[f"val/ce_overall_{key}"] = loss_sum / pos_count
        if mask_count > 0:
            metrics[f"val/ce_mask_{key}"] = mask_loss_sum / mask_count
        if clean_count > 0:
            metrics[f"val/ce_clean_{key}"] = clean_loss_sum / clean_count

    model.train()
    return metrics


# T3-D ADDED: per-step diagnostic metrics. Reads scalars off the model + optimiser to
# diagnose training pathologies (gate stuck, hidden states exploding, one param group
# drifting much faster than another). All ops are .item() on scalars or .norm() on
# already-existing param grads, so cost is negligible.
@torch.no_grad()
def _t3d_diagnostic_metrics(model, optimizer):
    metrics = {}

    # Resolve the underlying model (unwrap DDP/FSDP if needed).
    inner = getattr(model, "module", model)

    # Gate value at talk layer 0.
    try:
        anchor_cond = inner.talk_model.layers[0].anchor_conditioning
        if anchor_cond is not None:
            metrics["t3/gate"] = float(anchor_cond.gate_value)
            if anchor_cond.learnable and anchor_cond.alpha is not None:
                metrics["t3/alpha_raw"] = float(anchor_cond.alpha.item())
                if anchor_cond.alpha.grad is not None:
                    metrics["t3/alpha_grad"] = float(anchor_cond.alpha.grad.norm().item())
    except (AttributeError, IndexError):
        pass

    # Per-group grad norms -- useful when optimiser was split (Strategy C).
    try:
        for i, group in enumerate(optimizer.param_groups):
            grads = [p.grad for p in group["params"] if p.grad is not None]
            if grads:
                norm = torch.stack([g.detach().norm() for g in grads]).norm().item()
                metrics[f"t3/grad_norm_group{i}"] = norm
                metrics[f"t3/lr_group{i}"] = float(group["lr"])
    except Exception:
        pass

    return metrics


def main():
    dist.init_process_group(backend=get_nccl_backend())
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(
        dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager,
    )

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    # -------- Data (unchanged from DMax) --------------------------------------
    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)

    # T3-D ADDED: shared progress value for the step-based mask ramp. Workers read this
    # value (read-only, lock-free) to compute their per-sample sigma. The main process
    # updates `noise_progress.value` each training step. When the noise_range is a single
    # point (low == high), the ramp is a no-op and progress is not used.
    use_noise_ramp = args.data.noise_range_low != args.data.noise_range_high
    noise_progress = mp.Value("d", 0.0) if use_noise_ramp else None
    if use_noise_ramp:
        logger.info_rank0(
            f"[T3-D noise ramp] sigma linearly ramps from {args.data.noise_range_low} to "
            f"{args.data.noise_range_high} over the full training schedule."
        )

    if args.data.data_type == "conversation":
        if not tokenizer.chat_template:
            raise ValueError("No chat template found in the tokenizer.")
        transform = partial(
            process_mdm_sft_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            noise_range=(args.data.noise_range_low, args.data.noise_range_high),
            mask_token_id=args.train.mask_token_id,  # T3-D MODIFIED: configurable.
            progress_state=noise_progress,  # T3-D ADDED: step-based ramp (None disables it).
        )
    elif args.data.data_type == "tokenid":
        transform = partial(
            process_mdm_tokenized_example,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            noise_range=(args.data.noise_range_low, args.data.noise_range_high),
            mask_token_id=args.train.mask_token_id,  # T3-D MODIFIED: configurable.
        )
    else:
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

    if args.data.dataloader_type == "native":
        if args.data.datasets_type == "iterable":
            train_dataset = build_iterative_dataset(args.data.train_path, transform=transform, seed=args.train.seed)
        elif args.data.datasets_type == "mapping":
            train_dataset = build_mapping_dataset(args.data.train_path, transform=transform)
        elif args.data.datasets_type == "local":
            train_dataset = build_local_dataset(args.data.train_path, transform=transform)

        dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
        if args.data.datasets_type in ("mapping", "local"):
            dataset_length = dataset_length / args.train.data_parallel_size
        args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)

        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    # -------- Model (unchanged from DMax; the registered class is now ThinkTalkLLaDA2) --
    logger.info_rank0("Prepare model")
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
        force_use_huggingface=args.model.force_use_huggingface,
    )
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

    # T3-D ADDED: depth-scaled init for the talk transformer. VeOmni's load_model_weights
    # (already run above) uses uniform std=initializer_range for unmatched-key params,
    # which is wrong for from-scratch transformers (output projections must be scaled
    # 1/sqrt(2*(layer+1)) to keep residual variance from growing with depth). Must be
    # called BEFORE build_parallelize_model wraps the model in DDP/FSDP.
    if hasattr(model, "init_talk_layers_depth_scaled"):
        model.init_talk_layers_depth_scaled()
        logger.info_rank0(
            "[T3-D] applied depth-scaled init to talk layers (Megatron/GPT-NeoX recipe)."
        )

    get_optimizer_pre_hook = getattr(model, "get_optimizer_pre_hook", None)
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0,
    )

    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        betas=(args.train.beta1, args.train.beta2),
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
    )

    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(model, model_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    # T3-D ADDED: Strategy C -- differential LR. If lr_lm_head_ratio != 1.0, split the
    # optimizer's single param group into two: talk + alpha + anchor at args.train.lr,
    # lm_head at args.train.lr * lr_lm_head_ratio. The LambdaLR scheduler built below
    # multiplies its [0,1] step factor against each group's initial_lr, so both groups
    # warmup/decay on the same shape but with different peaks.
    if args.train.lr_lm_head_ratio != 1.0 and args.train.t3_rollout_mode != "none":
        lmhead_params = [
            p for n, p in model.named_parameters()
            if n.startswith("lm_head") and p.requires_grad
        ]
        lmhead_param_ids = {id(p) for p in lmhead_params}
        if not lmhead_params:
            logger.info_rank0(
                "[T3-D differential LR] lr_lm_head_ratio set but no trainable lm_head params "
                "found (train_lm_head likely false). Skipping split."
            )
        else:
            # Move LM head params out of group 0, into a new group with lower LR.
            optimizer.param_groups[0]["params"] = [
                p for p in optimizer.param_groups[0]["params"]
                if id(p) not in lmhead_param_ids
            ]
            optimizer.param_groups[0]["initial_lr"] = args.train.lr   # explicit for LambdaLR base_lrs

            lr_lmhead = args.train.lr * args.train.lr_lm_head_ratio
            optimizer.add_param_group({
                "params": lmhead_params,
                "lr": lr_lmhead,
                "initial_lr": lr_lmhead,
            })
            logger.info_rank0(
                f"[T3-D differential LR] talk+alpha+anchor: lr={args.train.lr:.2e}, "
                f"lm_head: lr={lr_lmhead:.2e} (ratio={args.train.lr_lm_head_ratio}). "
                f"Split: {sum(p.numel() for p in optimizer.param_groups[0]['params']):,} "
                f"non-lmhead params + {sum(p.numel() for p in lmhead_params):,} lm_head params."
            )

    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},
            )
        model_assets = [model_config, tokenizer]
        save_model_assets(args.train.model_assets_dir, model_assets)

    if args.train.profile_this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
    )

    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}
        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:
            iter(train_dataloader)
        dist.barrier()
        logger.info_rank0(f"Loaded distributed checkpoint from {args.train.load_checkpoint_path}")

    # -------- Build block diffusion attention mask (unchanged from DMax) ------
    if args.train.block_diffusion_mode:
        bd_attn_full_len = args.data.max_seq_len * 2
        block_size = args.train.block_size
        block_diffusion_attn_mask_flag = block_diffusion_mask(
            b=None, h=None,
            q_idx=torch.arange(bd_attn_full_len)[:, None],
            kv_idx=torch.arange(bd_attn_full_len)[None, :],
            block_size=block_size,
            n=args.data.max_seq_len,
        ).unsqueeze(0).unsqueeze(0)

        block_diffusion_attn_mask_prototype = torch.zeros_like(
            block_diffusion_attn_mask_flag,
            dtype=torch.float32 if args.train.enable_mixed_precision else torch.bfloat16,
        )
        block_diffusion_attn_mask_prototype.masked_fill_(block_diffusion_attn_mask_flag.logical_not(), float("-inf"))

        # T3-D ADDED: 3L mask + position_ids for the concat_segment talk pathway.
        # Built once at startup, reused per micro-batch. None-valued when model is in
        # gated_residual mode (cheap memory savings, also signals to .forward() that
        # the concat_segment path is not in use).
        injection_mode = getattr(model_config, "anchor_injection_mode", "gated_residual")
        use_concat_segment = injection_mode == "concat_segment"
        use_hybrid_xattn = injection_mode == "hybrid_xattn"
        mask_dtype = torch.float32 if args.train.enable_mixed_precision else torch.bfloat16

        if use_concat_segment:
            bd_attn_full_len_3L = args.data.max_seq_len * 3
            block_diffusion_attn_mask_flag_3L = block_diffusion_mask_3L(
                q_idx=torch.arange(bd_attn_full_len_3L)[:, None],
                kv_idx=torch.arange(bd_attn_full_len_3L)[None, :],
                block_size=block_size,
                n=args.data.max_seq_len,
            ).unsqueeze(0).unsqueeze(0)
            block_diffusion_attn_mask_prototype_3L = torch.zeros_like(
                block_diffusion_attn_mask_flag_3L,
                dtype=mask_dtype,
            )
            block_diffusion_attn_mask_prototype_3L.masked_fill_(
                block_diffusion_attn_mask_flag_3L.logical_not(), float("-inf"),
            )
            logger.info_rank0(
                f"[T3-D concat_segment] built 3L attention mask "
                f"({bd_attn_full_len_3L}x{bd_attn_full_len_3L})."
            )
        else:
            block_diffusion_attn_mask_prototype_3L = None

        # T3-D ADDED: L self-attn mask + L-by-2L cross-attn mask for the hybrid_xattn pathway.
        if use_hybrid_xattn:
            L_full = args.data.max_seq_len
            # Talk self-attn (L x L): block-diagonal among noisy positions only.
            self_attn_flag_L = talk_self_attn_mask_L(
                q_idx=torch.arange(L_full)[:, None],
                kv_idx=torch.arange(L_full)[None, :],
                block_size=block_size,
            ).unsqueeze(0).unsqueeze(0)
            talk_self_attn_mask_prototype_L = torch.zeros_like(self_attn_flag_L, dtype=mask_dtype)
            talk_self_attn_mask_prototype_L.masked_fill_(self_attn_flag_L.logical_not(), float("-inf"))

            # Talk cross-attn (L x 2L): noisy Q -> anchor[noisy half all, clean half block-causal].
            cross_attn_flag = talk_cross_attn_mask(
                q_idx=torch.arange(L_full)[:, None],
                kv_idx=torch.arange(2 * L_full)[None, :],
                block_size=block_size,
                n=L_full,
            ).unsqueeze(0).unsqueeze(0)
            talk_cross_attn_mask_prototype = torch.zeros_like(cross_attn_flag, dtype=mask_dtype)
            talk_cross_attn_mask_prototype.masked_fill_(cross_attn_flag.logical_not(), float("-inf"))

            logger.info_rank0(
                f"[T3-D hybrid_xattn] built talk self-attn mask ({L_full}x{L_full}) and "
                f"cross-attn mask ({L_full}x{2 * L_full})."
            )
        else:
            talk_self_attn_mask_prototype_L = None
            talk_cross_attn_mask_prototype = None

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit,
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, "
        f"epochs: {args.train.num_train_epochs}"
    )

    # T3-D ADDED: surface key T3 knobs on rank 0 so config drift is obvious in logs.
    if args.train.global_rank == 0:
        logger.info_rank0(
            f"[T3-D] rollout_mode={args.train.t3_rollout_mode} "
            f"rollout_target={args.train.t3_rollout_target} "
            f"rollout_replace={args.train.t3_rollout_replace} "
            f"train_iterations={args.train.t3_train_iterations}"
        )

    # T3-D ADDED: rollout-flag ratio ramp. When low != high, each micro_batch's flag is
    # resampled to Bernoulli(threshold) with threshold ramping linearly across training.
    # When low == high, the dataset's flag value is used unchanged (DMax-default behaviour).
    use_rollout_ramp = (
        args.train.t3_rollout_ratio_low != args.train.t3_rollout_ratio_high
    )
    if use_rollout_ramp:
        logger.info_rank0(
            f"[T3-D rollout ramp] flag-True probability linearly ramps from "
            f"{args.train.t3_rollout_ratio_low} to {args.train.t3_rollout_ratio_high} "
            f"over the full training schedule."
        )

    # T3-D ADDED: inline-validation setup. Read the same train file, take the tail of
    # the seed-shuffled order as a deterministic held-out set, build a fixed-sigma
    # transform factory. Validation runs at step 0 (baseline) and every t3_val_every
    # steps. Tail of the shuffle is the chunk training will reach LAST -- so at any
    # training step before that, it's truly unseen.
    val_enabled = (
        args.train.t3_val_every > 0
        and args.train.global_rank == 0
        and args.data.data_type == "conversation"
    )
    val_raw_examples: List[Dict[str, Any]] = []
    val_indices: List[int] = []
    val_sigmas: List[float] = []
    if val_enabled:
        logger.info_rank0(
            f"[T3-D val] Loading val examples from {args.data.train_path} ..."
        )
        with open(args.data.train_path) as _vf:
            for _line in _vf:
                val_raw_examples.append(json.loads(_line))
        _n_val_total = len(val_raw_examples)
        _val_gen = torch.Generator().manual_seed(args.train.seed)
        _shuffled = torch.randperm(_n_val_total, generator=_val_gen).tolist()
        val_indices = _shuffled[-args.train.t3_val_tail:]
        val_sigmas = [float(s.strip()) for s in args.train.t3_val_sigmas.split(",") if s.strip()]
        logger.info_rank0(
            f"[T3-D val] tail-{args.train.t3_val_tail} of seed-shuffled "
            f"{_n_val_total} examples; sigmas={val_sigmas}; "
            f"every {args.train.t3_val_every} steps."
        )

    # Baseline validation (step 0, before any training).
    if val_enabled:
        logger.info_rank0("[T3-D val] Running baseline validation (step 0)...")
        _val_metrics = _run_validation(
            model, val_raw_examples, val_indices, val_sigmas,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            block_size=args.train.block_size if args.train.block_diffusion_mode else 32,
            mask_token_id=args.train.mask_token_id,
            text_keys=args.data.text_keys,
            block_diffusion_attn_mask_prototype=block_diffusion_attn_mask_prototype,
            block_diffusion_attn_mask_prototype_3L=block_diffusion_attn_mask_prototype_3L,
            talk_self_attn_mask_prototype_L=talk_self_attn_mask_prototype_L,
            talk_cross_attn_mask_prototype=talk_cross_attn_mask_prototype,
            device=get_device_type(),
        )
        logger.info_rank0(f"[T3-D val] step 0: {_val_metrics}")
        if args.train.use_wandb:
            wandb.log(_val_metrics, step=0)

    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, args.train.train_steps):
            global_step += 1

            # T3-D ADDED: advance the shared noise-progress value for the step-based mask
            # ramp. Workers read this lock-free; some staleness due to prefetch is fine.
            total_steps = max(args.train.train_steps * args.train.num_train_epochs, 1)
            step_progress = min(global_step / total_steps, 1.0)
            if noise_progress is not None:
                noise_progress.value = step_progress

            # T3-D ADDED: compute this step's rollout-flag threshold (used to override the
            # dataset's per-micro_batch flag value when the ramp is active).
            if use_rollout_ramp:
                rollout_threshold = (
                    args.train.t3_rollout_ratio_low
                    + (
                        args.train.t3_rollout_ratio_high
                        - args.train.t3_rollout_ratio_low
                    ) * step_progress
                )
            else:
                rollout_threshold = None

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            # T3-D ADDED: split loss logging by OPUT flag.
            #   flag=False -> "mask path"   : input is the standard masked sequence.
            #   flag=True  -> "rollout path": input has talk's argmax in place of [MASK]s.
            # The rollout path is structurally harder (talk's argmax is mostly wrong early),
            # so loss_rollout_path > loss_mask_path is expected at start. As talk improves
            # the gap should shrink -- a useful signal independent of the noise ramp.
            loss_mask_path_sum = 0.0
            loss_mask_path_n = 0
            loss_rollout_path_sum = 0.0
            loss_rollout_path_n = 0
            # T3-D ADDED: per-iter loss accumulators (A4 multi-iter training).
            # iter 0 is the first forward (raw masked input or first reveal at flag=True);
            # iter k is the forward after k reveals. With t3_train_iterations=1 only iter 0
            # is populated; with N>1 we get a curve per training step that shows whether
            # later iterations achieve lower loss (the actual hypothesis test for A4).
            n_iter_max = max(int(args.train.t3_train_iterations), 1)
            loss_per_iter_sum = [0.0 for _ in range(n_iter_max)]
            loss_per_iter_n = [0 for _ in range(n_iter_max)]
            synchronize()
            start_time = time.time()
            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("source_name", None)

                if args.train.block_diffusion_mode:
                    noisy_input_ids = micro_batch["noisy_input_ids"]
                    clean_input_ids = micro_batch["input_ids"]
                    batch_size = noisy_input_ids.shape[0]
                    full_input_ids = torch.cat([noisy_input_ids, clean_input_ids], dim=1)
                    noisy_position_ids = torch.arange(noisy_input_ids.shape[1], device=get_device_type(), dtype=torch.long)
                    clean_position_ids = torch.arange(clean_input_ids.shape[1], device=get_device_type(), dtype=torch.long)
                    position_ids = torch.cat([noisy_position_ids, clean_position_ids], dim=0).unsqueeze(0).expand(batch_size, -1).clone()
                    micro_batch["input_ids"] = full_input_ids
                    micro_batch["position_ids"] = position_ids
                    micro_batch["attention_mask"] = block_diffusion_attn_mask_prototype.expand(batch_size, -1, -1, -1)

                    # T3-D ADDED: attach 3L mask + position_ids for the concat_segment talk pathway.
                    # In that mode the model.forward will assemble [noisy, anchor, clean] in talk
                    # and consume these 3L tensors. In gated_residual mode these stay None.
                    if block_diffusion_attn_mask_prototype_3L is not None:
                        micro_batch["attention_mask_3L"] = (
                            block_diffusion_attn_mask_prototype_3L.expand(batch_size, -1, -1, -1)
                        )
                        # 3L position_ids: noisy positions [0..L-1], then anchor positions
                        # [0..L-1] (sharing intra-block offsets with noisy), then clean
                        # [0..L-1]. All three streams use the same intra-block positions;
                        # segment_embed inside the model distinguishes them.
                        micro_batch["position_ids_3L"] = torch.cat(
                            [noisy_position_ids, noisy_position_ids, clean_position_ids], dim=0,
                        ).unsqueeze(0).expand(batch_size, -1).clone()

                    # T3-D ADDED: attach L self-attn mask + L-by-2L cross-attn mask for
                    # the hybrid_xattn pathway. In that mode talk processes only L noisy
                    # tokens; cross-attn carries the anchor (full 2L) as K/V.
                    if talk_self_attn_mask_prototype_L is not None:
                        micro_batch["attention_mask_L"] = (
                            talk_self_attn_mask_prototype_L.expand(batch_size, -1, -1, -1)
                        )
                        micro_batch["position_ids_L"] = (
                            noisy_position_ids.unsqueeze(0).expand(batch_size, -1).clone()
                        )
                        micro_batch["cross_attention_mask"] = (
                            talk_cross_attn_mask_prototype.expand(batch_size, -1, -1, -1)
                        )
                        # KV positions for cross-attn: noisy half [0..L-1] then clean half
                        # [0..L-1] (DMax parallel-position convention -- same as the 2L
                        # position_ids used by think on the doubled sequence).
                        micro_batch["cross_position_ids"] = torch.cat(
                            [noisy_position_ids, clean_position_ids], dim=0,
                        ).unsqueeze(0).expand(batch_size, -1).clone()
                else:
                    micro_batch["attention_mask"] = None

                micro_batch = {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }

                labels = micro_batch.pop("labels", None)
                flag = micro_batch.pop("flag", None)  # T3-D MODIFIED: pop flag here (was kept in dict in DMax).
                # T3-D ADDED: cache the per-micro-batch flag for the split-loss bookkeeping below.
                # When the rollout-ratio ramp is active, override the dataset's flag with a
                # fresh Bernoulli draw at this step's ramped threshold.
                if rollout_threshold is not None:
                    flag_bool = torch.rand(1).item() < rollout_threshold
                else:
                    flag_bool = bool(flag.item()) if flag is not None else False
                noisy_len = noisy_input_ids.shape[1] if args.train.block_diffusion_mode else micro_batch["input_ids"].shape[1]

                # =====================================================================
                # T3-D v6e -- MULTI-ITER TRAINING (A4).
                #
                # Both flag=False (mask path) and flag=True (rollout path) run N talk
                # forwards with grad. Loss is the uniform mean of per-iter CE.
                #   think runs once at the start; anchor is reused across all iterations.
                #   Reveal between iters:
                #     mask path    -> DMax-uniform (per-block, left-to-right confident
                #                     prefix, threshold=0.5, with leftmost-mask fallback).
                #                     Matches DMax inference's reveal rule.
                #     rollout path -> full argmax (every still-masked position is replaced
                #                     with the model's argmax). Harder distribution; no
                #                     confidence gating.
                # gradients accumulate via backward-per-iter (memory bounded to one iter's
                # activations at a time).
                # =====================================================================

                # 1. Compute anchor once (no-grad). Reused by every iter.
                model.eval()
                with torch.no_grad():
                    anchor_cached = model.run_think_and_anchor(
                        input_ids=micro_batch["input_ids"],
                        attention_mask=micro_batch["attention_mask"],
                        position_ids=micro_batch["position_ids"],
                    )
                model.train()

                # 2. Pick reveal strategy per flag.
                reveal_kind = "full_argmax" if flag_bool else "dmax_uniform"

                # 3. N grad iterations. Each iter: forward, CE, backward, then no-grad reveal.
                n_iters = max(int(args.train.t3_train_iterations), 1)
                for iter_idx in range(n_iters):
                    with model_fwd_context:
                        logits = model.run_talk(
                            input_ids=micro_batch["input_ids"],
                            anchor=anchor_cached.detach(),
                            attention_mask=micro_batch["attention_mask"],
                            position_ids=micro_batch["position_ids"],
                            attention_mask_3L=micro_batch.get("attention_mask_3L"),
                            position_ids_3L=micro_batch.get("position_ids_3L"),
                            attention_mask_L=micro_batch.get("attention_mask_L"),
                            position_ids_L=micro_batch.get("position_ids_L"),
                            cross_attention_mask=micro_batch.get("cross_attention_mask"),
                            cross_position_ids=micro_batch.get("cross_position_ids"),
                        )

                        if args.train.block_diffusion_mode:
                            noisy_logits = logits[:, :noisy_len].contiguous()
                        else:
                            noisy_logits = logits

                        if args.train.same_token_labels:
                            unscaled_loss = torch.nn.functional.cross_entropy(
                                noisy_logits.view(-1, noisy_logits.shape[-1]),
                                labels.view(-1),
                                reduction="none",
                            )
                            valid = (labels != -100).sum().clamp_min(1)
                            ce_iter = unscaled_loss.sum() / valid
                        else:
                            shifted_logits = noisy_logits[:, :-1, :].contiguous()
                            shifted_labels = labels[:, 1:].contiguous()
                            unscaled_loss = torch.nn.functional.cross_entropy(
                                shifted_logits.view(-1, shifted_logits.shape[-1]),
                                shifted_labels.view(-1),
                                reduction="none",
                            )
                            valid = (shifted_labels != -100).sum().clamp_min(1)
                            ce_iter = unscaled_loss.sum() / valid

                        # Scale for grad accumulation across iters x micro_batches.
                        loss_scaled = ce_iter / (n_iters * len(micro_batches))

                    with model_bwd_context:
                        loss_scaled.backward()

                    # Per-iter / per-flag / total loss bookkeeping.
                    ce_iter_val = float(ce_iter.item())
                    total_loss += float(loss_scaled.item())   # sum gives mean CE across iter+mb
                    loss_per_iter_sum[iter_idx] += ce_iter_val
                    loss_per_iter_n[iter_idx] += 1
                    if flag_bool:
                        loss_rollout_path_sum += ce_iter_val
                        loss_rollout_path_n += 1
                    else:
                        loss_mask_path_sum += ce_iter_val
                        loss_mask_path_n += 1

                    # 4. Reveal for next iter (no-grad). Skip on last iter (no next forward).
                    if iter_idx < n_iters - 1:
                        with torch.no_grad():
                            current_noisy = micro_batch["input_ids"][:, :noisy_len]
                            if reveal_kind == "dmax_uniform":
                                new_noisy = reveal_dmax_uniform(
                                    logits=noisy_logits.detach(),
                                    current_noisy=current_noisy,
                                    mask_token_id=args.train.mask_token_id,
                                    block_size=args.train.block_size,
                                    threshold=args.train.t3_reveal_threshold,
                                )
                            else:  # full_argmax
                                new_noisy = reveal_full_argmax(
                                    logits=noisy_logits.detach(),
                                    current_noisy=current_noisy,
                                    mask_token_id=args.train.mask_token_id,
                                )
                            # Update the full input_ids; we replace the noisy half only.
                            new_input = micro_batch["input_ids"].clone()
                            new_input[:, :noisy_len] = new_noisy
                            micro_batch["input_ids"] = new_input

                del micro_batch

            # ---- Optimiser step (unchanged from DMax) ----------------------------
            if hasattr(model, "clip_grad_norm_"):
                _gn = model.clip_grad_norm_(args.train.max_grad_norm)
                grad_norm = _gn.item() if hasattr(_gn, "item") else float(_gn)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            total_loss, grad_norm = all_reduce(
                (total_loss, grad_norm), group=get_parallel_state().fsdp_group,
            )
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            data_loader_tqdm.set_postfix_str(f"loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}")
            data_loader_tqdm.update()

            if args.train.global_rank == 0 and args.train.use_wandb:
                train_metrics.update({
                    "training/loss": total_loss,
                    "training/grad_norm": grad_norm,
                    "training/lr": lr,
                })
                # T3-D ADDED: diagnostic logs to help spot training pathologies (gate
                # stuck closed, talk hidden norm exploding, lm_head drifting too fast).
                # All cheap to compute -- one .item() per step.
                t3_metrics = _t3d_diagnostic_metrics(model, optimizer)
                if t3_metrics:
                    train_metrics.update(t3_metrics)
                if noise_progress is not None:
                    progress = float(noise_progress.value)
                    sigma = (
                        args.data.noise_range_low
                        + (args.data.noise_range_high - args.data.noise_range_low) * progress
                    )
                    train_metrics["t3/noise_ramp_progress"] = progress
                    train_metrics["t3/noise_ramp_sigma"] = sigma
                # T3-D ADDED: per-flag loss split (mask path vs rollout path).
                # Only logged when this step actually contained micro_batches of that flag
                # (each global step typically contains both, since flag is randomised
                # per-sample, but we guard against the all-one-flag edge case).
                if loss_mask_path_n > 0:
                    train_metrics["training/loss_mask_path"] = (
                        loss_mask_path_sum / loss_mask_path_n
                    )
                if loss_rollout_path_n > 0:
                    train_metrics["training/loss_rollout_path"] = (
                        loss_rollout_path_sum / loss_rollout_path_n
                    )
                total_micro_n = loss_mask_path_n + loss_rollout_path_n
                if total_micro_n > 0:
                    # Note: with multi-iter, this rate is "iter-samples with flag=True"
                    # divided by total iter-samples (= n_iters x len(micro_batches)). Since
                    # the flag is per-micro_batch, this still recovers the per-micro_batch
                    # flag-True rate up to constant scaling.
                    train_metrics["t3/rollout_flag_rate"] = (
                        loss_rollout_path_n / total_micro_n
                    )
                if rollout_threshold is not None:
                    train_metrics["t3/rollout_ratio_target"] = rollout_threshold
                # T3-D ADDED: per-iter loss curve. With multi-iter training (v6e), each
                # step contributes one CE per iter per micro_batch; we log the mean CE for
                # each iter index. If multi-iter is working as intended, loss_iter_0 stays
                # near baseline while loss_iter_{k>0} drops as talk learns to use the
                # progressively-revealed input.
                for i in range(n_iter_max):
                    if loss_per_iter_n[i] > 0:
                        train_metrics[f"training/loss_iter_{i}"] = (
                            loss_per_iter_sum[i] / loss_per_iter_n[i]
                        )

            # T3-D ADDED: inline validation. Runs every t3_val_every steps on the
            # deterministic tail-N held-out subset, computing CE at fixed sigmas.
            # Always logs as `val/ce_*` on the same step as training metrics so wandb
            # plots them together.
            if val_enabled and global_step % args.train.t3_val_every == 0:
                _val_metrics = _run_validation(
                    model, val_raw_examples, val_indices, val_sigmas,
                    tokenizer=tokenizer,
                    max_seq_len=args.data.max_seq_len,
                    block_size=args.train.block_size if args.train.block_diffusion_mode else 32,
                    mask_token_id=args.train.mask_token_id,
                    text_keys=args.data.text_keys,
                    block_diffusion_attn_mask_prototype=block_diffusion_attn_mask_prototype,
                    block_diffusion_attn_mask_prototype_3L=block_diffusion_attn_mask_prototype_3L,
                    talk_self_attn_mask_prototype_L=talk_self_attn_mask_prototype_L,
                    talk_cross_attn_mask_prototype=talk_cross_attn_mask_prototype,
                    device=get_device_type(),
                )
                logger.info_rank0(f"[T3-D val] step {global_step}: {_val_metrics}")
                if args.train.use_wandb and args.train.global_rank == 0:
                    wandb.log(_val_metrics, step=global_step)
                wandb.log(train_metrics, step=global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()

            # ---- Checkpoint save (unchanged from DMax) ---------------------------
            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path}")

                if args.train.global_rank == 0 and args.train.save_hf_weights:
                    try:
                        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
                        helper.empty_cache()
                        model_state_dict = ckpt_to_state_dict(
                            save_checkpoint_path=save_checkpoint_path,
                            output_dir=args.train.output_dir,
                            ckpt_manager=args.train.ckpt_manager,
                        )
                        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
                        logger.info_rank0(f"HF checkpoint saved at {hf_weights_path}")
                        del model_state_dict
                        helper.empty_cache()
                    except Exception as e:
                        logger.info_rank0(f"Failed to save HF checkpoint: {e}")
                dist.barrier()

        data_loader_tqdm.close()
        start_step = 0
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path}")

    synchronize()
    del optimizer, lr_scheduler
    helper.empty_cache()
    if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            output_dir=args.train.output_dir,
            ckpt_manager=args.train.ckpt_manager,
        )
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"HF checkpoint saved at {hf_weights_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
