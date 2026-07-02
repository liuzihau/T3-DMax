# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
"""DMax HEAVY fine-tuner (OPUT with draft-contaminated rollout) — copied from `train_dbet.py` to reuse the
VeOmni framework + the dual-stream data feeding. Unlike train_dbet (frozen heavy, trainable drafter), this
FREEZES the drafter and fine-tunes only the heavy's TOP layers (`trainable_from_layer:`), so the heavy learns
to handle the drafter's top-k soft-embeds and correct them toward gold:
  - freeze all but `heavy.model.layers[trainable_from_layer:]` (set before FSDP; build_optimizer filters);
  - core step = `dmax_dbet_train_core.dmax_dbet_train_step` (merged 2-route: heavy fwd#1 -> mask-denoise loss_B
    + heavy_commit -> draft fill (top-k soft-embed) -> heavy fwd#2 -> draft-correct loss_A);
  - mask path unchanged (dual-stream block-diffusion mask); drafter held-out eval is OFF (eval heavy-only GSM8K
    offline via dInfer). REQUIRES `--block_diffusion_mode true`. Assumes micro_batch_size 1 (drafter gather)."""

import json
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
ModelRegistry.register_modeling_path("models.dbet")   # model class resolved via architectures=[DbetForDraftDecoding]
from transformers import AutoConfig, AutoModelForCausalLM
from models.dbet import DbetConfig, DbetForDraftDecoding
AutoConfig.register(DbetConfig.model_type, DbetConfig)   # resolve config.json model_type ("dbet_veomni") -> DbetConfig
AutoModelForCausalLM.register(DbetConfig, DbetForDraftDecoding)
from dataset.data_transform_dbet import process_mdm_tokenized_example, process_mdm_sft_example
from dataset import build_local_dataset
from dmax_dbet_train_core import dmax_dbet_train_step
from dbet_metrics import MetricsLogger  # (drafter eval is off for the heavy fine-tune)
import random


logger = helper.create_logger(__name__)

@dataclass
class LLaDA2ModelArguments(ModelArguments):
    attn_implementation: Optional[Literal["eager", "sdpa", "flex_attention"]] = field(
        default="sdpa",
        metadata={"help": "Attention implementation to use."},
    )


@dataclass
class LLaDA2DataArguments(DataArguments):
    data_type: Literal["conversation", "tokenid"] = field(
        default="conversation",
        metadata={"help": "Type of the training data."},
    )
    datasets_type: Literal["mapping", "local"] = field(
        default="mapping",
        metadata={"help": "Type of the datasets."},
    )
    text_keys: str = field(
        default="messages",
        metadata={"help": "Key to get text from the training data."},
    )
    noise_range_low: float = field(
        default=0.3,
        metadata={"help": "Noise level for random flip input_ids to mask_ids"}
    )
    noise_range_high: float = field(
        default=0.8,
        metadata={"help": "Noise level for random flip input_ids to mask_ids"}
    )
    revealed_corrupt_rate: float = field(
        default=0.0,
        metadata={"help": "DBet (EAGLE-style aug): prob of corrupting each REVEALED answer token to a random "
                          "token in the NOISY stream, so the heavy hidden is imperfect and the drafter must "
                          "reconstruct (mimics heavy-decoded context at inference). 0 disables; try ~0.05-0.15."}
    )

    def __post_init__(self):
        super().__post_init__()
        if self.noise_range_low > self.noise_range_high:
            raise ValueError(
                f"noise_range_low ({self.noise_range_low}) "
                f"cannot be greater than noise_range_high ({self.noise_range_high})."
            )

        if not (0.0 <= self.noise_range_low <= 1.0):
            raise ValueError(
                f"noise_range_low must be between 0.0 and 1.0, but got {self.noise_range_low}."
            )

        if not (0.0 <= self.noise_range_high <= 1.0):
            raise ValueError(
                f"noise_range_high must be between 0.0 and 1.0, but got {self.noise_range_high}."
            )


@dataclass
class LLaDA2TrainingArguments(TrainingArguments):
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW optimizer beta1."},
    )
    beta2: float = field(
        default=0.999,
        metadata={"help": "AdamW optimizer beta2"},
    )
    block_diffusion_mode: bool = field(
        default=False,
        metadata={"help": "If train MDM in block_diffusion mode. True: use block_diffusion, False: full_attention"}
    )
    block_size: int = field(
        default=32,
        metadata={"help": "The block size for block diffusion block size"}
    )
    same_token_labels: bool = field(
        default=False,
        metadata={"help": "If use same token location labels. True: no shift, False: use next-token prediction shift."}
    )
    heavy_commit_threshold: float = field(
        default=0.9,
        metadata={"help": "DBet: confidence threshold for the frozen heavy's one-pass decode_uniform commit "
                          "(left-to-right prefix until conf < threshold). Higher = heavy commits less, drafter does more."}
    )
    conf_loss_weight: float = field(
        default=1.0,
        metadata={"help": "DBet: weight of the confidence-head BCE relative to the token CE."}
    )
    # --- DMax heavy fine-tune (train_dmax_dbet_oput) ---
    trainable_from_layer: int = field(
        default=8,
        metadata={"help": "Fine-tune heavy.model.layers[from:] (top layers); freeze below + embed/lm_head + "
                          "the whole drafter. 8 -> top 12 of 20 (2 H200 fit)."}
    )
    heavy_commit_threshold_low: float = field(default=0.75, metadata={"help": "heavy commit thr sampled in [lo,hi]."})
    heavy_commit_threshold_high: float = field(default=0.9, metadata={"help": "heavy commit thr sampled in [lo,hi]."})
    draft_top_k_choices: str = field(default="1,2,3", metadata={"help": "draft soft-embed top-k sampled per example."})
    heavy_soft_top_k: int = field(default=1, metadata={"help": "heavy-committed soft-embed top-k (DMax default 1)."})
    heavy_soft_tau: float = field(default=1.0, metadata={"help": "heavy-committed soft-embed temperature."})
    draft_soft_tau: float = field(default=1.0, metadata={"help": "draft-committed soft-embed temperature."})
    loss_b_weight: float = field(default=1.0, metadata={"help": "weight of the mask-denoise loss (Route B)."})
    loss_a_weight: float = field(default=1.0, metadata={"help": "weight of the draft-correct loss (Route A)."})
    eval_heavy_thr: float = field(default=0.8, metadata={"help": "held-out val: FIXED heavy commit threshold "
                                                                 "(both the corr2 commit and the acc_heavy3 decode)."})
    eval_draft_k: int = field(default=2, metadata={"help": "held-out val: FIXED draft soft-embed top-k."})
    eval_n_heavy_passes: int = field(default=3, metadata={"help": "held-out val: heavy passes for acc_heavy3 "
                                                                  "(matched to acc_corr2's 2 heavy + 1 draft)."})
    loss_decay_mode: str = field(
        default="dbet",
        metadata={"help": "DBet loss-weight schedule over remaining positions: 'dbet' (max(base^k,floor); "
                          "original), 'dbet_twophase' (gentle head^k for first `window`, then steep tail decay), "
                          "or 'dflash' (exp(-k/gamma), DFlash Eq.4)."}
    )
    loss_decay_base: float = field(default=0.9, metadata={"help": "'dbet' per-position decay base."})
    loss_decay_floor: float = field(default=0.1, metadata={"help": "Minimum weight floor (try 0.025 for twophase)."})
    loss_decay_head: float = field(default=0.95, metadata={"help": "'dbet_twophase' gentle decay for k<window."})
    loss_decay_tail: float = field(default=0.8, metadata={"help": "'dbet_twophase' steep decay for k>=window."})
    loss_decay_window: int = field(default=6, metadata={"help": "'dbet_twophase' head/tail boundary (first-T)."})
    loss_decay_gamma: float = field(default=14.0, metadata={"help": "'dflash' decay rate gamma (~block_size/2)."})
    log_steps: int = field(
        default=10,
        metadata={"help": "DBet: write a train-metrics record (loss/tok/conf/acc/grad_norm/lr/tok_s) every N steps."}
    )
    eval_steps: int = field(
        default=0,
        metadata={"help": "DBet: run held-out validation (sigma sweep) every N steps. 0 disables eval."}
    )
    eval_holdout_size: int = field(
        default=128,
        metadata={"help": "DBet: number of tail examples of the train file held out for validation."}
    )
    eval_sigmas: str = field(
        default="0.1,0.3,0.5,0.7,0.9",
        metadata={"help": "DBet: comma-separated mask ratios swept during eval (acc/AUC-vs-mask-ratio)."}
    )
    eval_tpf: int = field(
        default=6,
        metadata={"help": "DBet: heavy tokens-per-forward window. acc6/top*_6/h2_acc6 average over the first "
                          "eval_tpf remaining positions per block (the decision-relevant span)."}
    )
    eval_heavy_second: bool = field(
        default=True,
        metadata={"help": "DBet: also run a heavy SECOND forward pass in eval (re-forward on the committed "
                          "sequence) to report h2_acc/h2_acc6 -- the teacher ceiling. ~2x eval cost."}
    )
    eval_at_start: bool = field(
        default=True,
        metadata={"help": "DBet: also run one eval before training (step 0 baseline) for the figures."}
    )
    metrics_path: str = field(
        default="",
        metadata={"help": "DBet: JSONL metrics path. Empty -> <output_dir>/dbet_metrics.jsonl."}
    )
    skip_nonfinite_steps: bool = field(
        default=True,
        metadata={"help": "DBet: skip the optimizer step when grad_norm is NaN/Inf (bf16 stability guard) so a "
                          "single overflow can't permanently poison the weights."}
    )


@dataclass
class Arguments:
    model: "LLaDA2ModelArguments" = field(default_factory=LLaDA2ModelArguments)
    data: "LLaDA2DataArguments" = field(default_factory=LLaDA2DataArguments)
    train: "LLaDA2TrainingArguments" = field(default_factory=LLaDA2TrainingArguments)


def block_diffusion_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
    """
    Constructs the specialized block diffusion attention mask for training
    composed of three masks:
    - **Block Diagonal Mask (M_BD)**: Self-attention within noised blocks
    - **Offset Block Causal Mask (M_OBC)**: Cross-attention for conditional context
    - **Block Causal Mask (M_BC)**: Attention to update x0

    Args:
        b, h: Batch and head indices (ignored for mask logic).
        q_idx, kv_idx: Query and Key indices.
        seq_len: Total sequence length.
        block_size: Defines the block structure.

    Returns:
        A boolean attention mask.
    """

    # Indicate whether token belongs to xt or x0
    x0_flag_q = (q_idx >= n)
    x0_flag_kv = (kv_idx >= n)

    # Compute block indices
    block_q = torch.where(x0_flag_q == 1,
                          (q_idx - n) // block_size,
                          q_idx // block_size)
    block_kv = torch.where(x0_flag_kv == 1,
                           (kv_idx - n) // block_size,
                           kv_idx // block_size)

    # **1. Block Diagonal Mask (M_BD) **
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # **2. Offset Block-Causal Mask (M_OBC) **
    offset_block_causal = (
        (block_q > block_kv)
        & (x0_flag_kv == 1)
        & (x0_flag_q == 0)
    )

    # **3. Block-Causal Mask (M_BC) **
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    # **4. Combine Masks **
    return block_diagonal | offset_block_causal | block_causal


def _drafter_only_state_dict(state_dict):
    """Drop the frozen heavy (heavy.*) from an exported state dict -- it's the DMax-Math-16B, fully recoverable
    from its source checkpoint, so there's no reason to duplicate ~16B params in every DBet export. Keep only the
    trained drafter (draft.*). To reload for inference: rebuild the heavy from the DMax source and
    `DbetForDraftDecoding(cfg, _heavy=heavy).load_state_dict(this, strict=False)` (same pattern as build_dbet_init)."""
    return {k: v for k, v in state_dict.items() if not k.startswith("heavy.")}


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

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

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

    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    # EAGLE-style robustness aug: corrupt a fraction of revealed answer tokens (noisy stream only).
    _corrupt_rate = getattr(args.data, "revealed_corrupt_rate", 0.0)
    _vocab_size = len(tokenizer) if _corrupt_rate > 0 else None
    if _corrupt_rate > 0:
        logger.info_rank0(f"Revealed-token corruption ON: rate={_corrupt_rate}, vocab={_vocab_size} (noisy stream).")
    if args.data.data_type == "conversation":
        if not tokenizer.chat_template:
            raise ValueError(f"No chat template found in the tokenizer.")

        transform = partial(
            process_mdm_sft_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            block_size=args.train.block_size,   # DBet: left-to-right per-block reveal
            text_keys=args.data.text_keys,
            noise_range=(args.data.noise_range_low, args.data.noise_range_high),
            mask_token_id=156895,
            corrupt_rate=_corrupt_rate, vocab_size=_vocab_size,
        )
    elif args.data.data_type == "tokenid":
        transform = partial(
            process_mdm_tokenized_example,
            max_seq_len=args.data.max_seq_len,
            block_size=args.train.block_size,   # DBet: left-to-right per-block reveal
            text_keys=args.data.text_keys,
            noise_range=(args.data.noise_range_low, args.data.noise_range_high),
            mask_token_id=156895,
            corrupt_rate=_corrupt_rate, vocab_size=_vocab_size,
        )
    else:
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}.")

    if args.data.dataloader_type == "native":
        if args.data.datasets_type == "iterable":
            logger.info_rank0("Start building iterative dataset")
            train_dataset = build_iterative_dataset(args.data.train_path, transform=transform, seed=args.train.seed)
        elif args.data.datasets_type == "mapping":
            logger.info_rank0("Start building mapping dataset")
            train_dataset = build_mapping_dataset(args.data.train_path, transform=transform)
        elif args.data.datasets_type == "local":
            logger.info_rank0("Start building local dataset")
            train_dataset = build_local_dataset(args.data.train_path, transform=transform)
        
        dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
        if args.data.datasets_type == "mapping" or args.data.datasets_type == "local":
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

    logger.info_rank0("Prepare model")
    import logging as _logging   # silence VeOmni's per-key weight-load/broadcast INFO spam (fills the screen)
    _logging.getLogger("veomni.models.module_utils").setLevel(_logging.WARNING)
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

    # ---- HEAVY FINE-TUNE freeze: train only heavy.model.layers[trainable_from_layer:] (top layers); freeze the
    # rest of the heavy + embed/lm_head + the whole drafter. build_optimizer filters requires_grad. Set BEFORE FSDP.
    _from = args.train.trainable_from_layer
    for _p in model.parameters():
        _p.requires_grad_(False)
    _n_layers = len(model.heavy.model.layers)
    for _layer in model.heavy.model.layers[_from:]:
        for _p in _layer.parameters():
            _p.requires_grad_(True)
    _n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info_rank0(f"[dmax-ft] training heavy layers [{_from}:{_n_layers}] "
                      f"({_n_layers - _from} layers, {_n_train/1e9:.2f}B params); everything else frozen.")

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
        broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0
    )

    # On 1-GPU/non-FSDP the model is returned unwrapped, so the FSDP clip_grad_norm_ is never registered and the
    # train loop would log "Can NOT find regitsered clip_grad_norm_" every step. Attach the standard clip so the
    # hasattr branch is taken silently (identical numerics to the fallback).
    if not hasattr(model, "clip_grad_norm_"):
        import types
        model.clip_grad_norm_ = types.MethodType(
            lambda self, max_norm: torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm), model)

    # NOTE: unlike the 1-GPU train_dbet, we do NOT manually cast the fused experts to bf16 here. Under FSDP2
    # mixed precision, FSDP's MixedPrecision casts ALL params to bf16 for compute (so the group_gemm kernel gets
    # bf16 experts automatically), and FSDP requires UNIFORM param dtype per shard unit -- a manual bf16 cast of
    # only the experts makes the model mixed fp32/bf16 and trips "FSDP expects uniform original parameter dtype".

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
                config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
            )

        # save model_assets before training
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
    nonfinite_skips = 0
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
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    # Build block diffusion attention mask
    if args.train.block_diffusion_mode:
        bd_attn_full_len = args.data.max_seq_len * 2
        block_size = args.train.block_size
        # NOTE: Boolean dtype block diffusion attention mask
        block_diffusion_attn_mask_flag = block_diffusion_mask(
            b=None, h=None,
            q_idx=torch.arange(bd_attn_full_len)[:, None],
            kv_idx=torch.arange(bd_attn_full_len)[None, :],
            block_size=block_size,
            n=args.data.max_seq_len
        ).unsqueeze(0).unsqueeze(0)
        
        block_diffusion_attn_mask_prototype = torch.zeros_like(
            block_diffusion_attn_mask_flag, 
            dtype=torch.float32 if args.train.enable_mixed_precision else torch.bfloat16
        )
        block_diffusion_attn_mask_prototype.masked_fill_(block_diffusion_attn_mask_flag.logical_not(), float("-inf"))

    # ---- train-metrics logging (drafter held-out eval does NOT apply to the heavy fine-tune) ----
    metrics_path = args.train.metrics_path or os.path.join(args.train.output_dir, "dmax_ft_metrics.jsonl")
    metrics_logger = MetricsLogger(
        jsonl_path=metrics_path if args.train.global_rank == 0 else "",
        use_wandb=args.train.use_wandb and args.train.global_rank == 0,
        enabled=args.train.global_rank == 0,
    )

    # ---- held-out val: base-skill guard (acc_heavy1), draft-correct (acc_corr2, 2 heavy+draft), matched-budget
    #      pure-heavy (acc_heavy3, 3 heavy @eval_heavy_thr), + loss_B/loss_A. Same step through model(...) under
    #      no_grad on the last-N train examples, FIXED reveal/th/draft_k (stable curve). COLLECTIVE: every rank
    #      runs model(...) in sync (FSDP all-gather); only rank0 logs. Deterministic -> ranks agree.
    from dbet_metrics import load_holdout_examples as _load_holdout, _build_dual_stream as _dual
    from dataset.data_transform_dbet import block_left_to_right_reveal as _reveal
    from dbet_train_core import MASK_ID as _MASK_ID
    _ft_holdout = None
    if args.train.eval_steps:
        _ft_holdout = _load_holdout(args.data.train_path, args.train.eval_holdout_size, tokenizer,
                                    args.data.max_seq_len, args.data.text_keys)
        logger.info_rank0(f"[dmax-ft eval] {len(_ft_holdout)} held-out examples, th={args.train.eval_heavy_thr} "
                          f"draft_k={args.train.eval_draft_k} n_heavy={args.train.eval_n_heavy_passes} -> {metrics_path}")

    def _run_eval(step):
        if not _ft_holdout:
            return
        was_training = model.training
        model.eval()
        L, bs, dev = args.data.max_seq_len, args.train.block_size, get_device_type()
        agg, keys = {}, ("loss_B", "loss_A", "acc_heavy1", "acc_corr2", "acc_heavy3")
        for clean_ids, prompt_len in _ft_holdout:
            clean_ids = clean_ids[:L]
            maskable = torch.arange(L) >= prompt_len
            noisy = _reveal(clean_ids.clone(), (args.data.noise_range_low, args.data.noise_range_high),
                            maskable, _MASK_ID, bs)
            mb = _dual(noisy.unsqueeze(0), clean_ids.unsqueeze(0), block_diffusion_attn_mask_prototype, dev)
            m = model(dmax_ft_eval_kwargs=dict(
                full=mb["input_ids"], attention_mask=mb["attention_mask"], position_ids=mb["position_ids"],
                noisy_len=L, mask_id=_MASK_ID, heavy_thr=args.train.eval_heavy_thr, draft_k=args.train.eval_draft_k,
                heavy_top_k=int(args.train.heavy_soft_top_k), heavy_tau=float(args.train.heavy_soft_tau),
                draft_tau=float(args.train.draft_soft_tau), block_size=bs, n_heavy_passes=args.train.eval_n_heavy_passes))
            for k in keys:
                agg[k] = agg.get(k, 0.0) + float(m[k])
        n = max(1, len(_ft_holdout))
        scalar = {k: agg[k] / n for k in keys}
        metrics_logger.log(scalar, step=step, split="val")
        logger.info_rank0(f"[dmax-ft eval @ {step}] " + " ".join(f"{k}={v:.4f}" for k, v in scalar.items())
                          + "   (want acc_corr2 >= acc_heavy3, acc_heavy1 steady)")
        if was_training:
            model.train()
        helper.empty_cache()

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
    if args.train.eval_steps and args.train.eval_at_start:
        _run_eval(global_step)   # step-0 baseline (untrained drafter) for the figures
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

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            step_m = {}                                     # accumulated DBet metrics (tok/conf/acc/acc6)
            # Only compute the extra train metrics (acc/acc6 + the .item() syncs) on steps we actually log --
            # saves per-step host<->device syncs. The training loss itself is computed every step regardless.
            want_metrics = (args.train.global_rank == 0 and global_step % max(1, args.train.log_steps) == 0)
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
                else:
                    micro_batch["attention_mask"] = None

                micro_batch = {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }

                micro_batch.pop("labels", None)   # gold = the clean stream

                # ===== DMax heavy fine-tune step (merged 2-route: mask-denoise + draft-correct) =====
                # heavy fwd#1 (grad) -> loss_B + commit -> draft fill (top-k soft-embed) -> heavy fwd#2 (grad) ->
                # loss_A. Only the heavy's top layers train; the drafter is frozen. Mask path unchanged.
                with model_fwd_context:
                    out = dmax_dbet_train_step(model, micro_batch, len(micro_batches), args, return_metrics=want_metrics)
                loss = out[0] if want_metrics else out

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                if want_metrics:
                    for _k, _v in out[1].items():
                        step_m[_k] = step_m.get(_k, 0.0) + float(_v)
                del micro_batch

            # Prefer model-provided clip_grad_norm_ (now both FSDP1 and FSDP2 registers custom grad norm clipping)
            if hasattr(model, "clip_grad_norm_"):
                _gn = model.clip_grad_norm_(args.train.max_grad_norm)
                grad_norm = _gn.item() if hasattr(_gn, "item") else float(_gn)
            else:
                logger.info_rank0(
                    "Can NOT find regitsered clip_grad_norm_ method in the model, using PyTorch default implementation.."
                )
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm)

            # Skip non-finite steps: in bf16 (1-GPU, no fp32 master) a gradient spike can overflow; without this
            # guard the next optimizer.step() writes NaN into the weights and training never recovers.
            _gn_val = grad_norm.full_tensor().item() if hasattr(grad_norm, "full_tensor") else float(grad_norm)
            _gn_finite = _gn_val == _gn_val and abs(_gn_val) != float("inf")
            if args.train.skip_nonfinite_steps and not _gn_finite:
                nonfinite_skips += 1
                optimizer.zero_grad()
                if nonfinite_skips <= 20 or nonfinite_skips % 100 == 0:
                    logger.info_rank0(f"[DBet] skipped non-finite step (grad_norm={_gn_val}) at step "
                                      f"{global_step}; total skips={nonfinite_skips}")
                lr_scheduler.step()           # keep schedule aligned with global_step
            else:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            data_loader_tqdm.set_postfix_str(f"loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}")
            data_loader_tqdm.update()

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update(
                        {"training/loss": total_loss, "training/grad_norm": grad_norm, "training/lr": lr}
                    )
                    wandb.log(train_metrics, step=global_step)

                # DBet train-metrics record (JSONL + wandb via MetricsLogger), every log_steps
                if global_step % max(1, args.train.log_steps) == 0:
                    nmb = max(1, len(micro_batches))
                    rec = {"loss": float(total_loss), "grad_norm": float(grad_norm), "lr": float(lr)}
                    rec.update({k: v / nmb for k, v in step_m.items()})   # tok, conf, acc, n_remaining (mean/micro-batch)
                    metrics_logger.log(rec, step=global_step, split="train")

            # DBet held-out validation (sigma sweep) every eval_steps
            if args.train.eval_steps and global_step % args.train.eval_steps == 0:
                _run_eval(global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()

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
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")


                # This code block is inside the if statement, so the HF ckpt is converted and saved immediately after saving the original ckpt
                if args.train.global_rank == 0 and args.train.save_hf_weights:
                    try:
                        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
                        
                        # Clear VRAM/RAM to prevent OOM during the conversion process
                        helper.empty_cache()
                        
                        logger.info_rank0(f"Converting to HF weights at {hf_weights_path}...")
                        
                        # Perform the conversion
                        model_state_dict = ckpt_to_state_dict(
                            save_checkpoint_path=save_checkpoint_path,
                            output_dir=args.train.output_dir,
                            ckpt_manager=args.train.ckpt_manager,
                        )
                        # HEAVY fine-tune: KEEP the full model (fine-tuned heavy.* + frozen draft.*) so hf_ckpt is
                        # a self-contained, directly-loadable DBet model. (train_dbet dropped heavy.* because the
                        # heavy was frozen there; here the heavy IS what we trained -- dropping it loses the run.)
                        logger.info_rank0(f"HF export: {len(model_state_dict)} tensors (full model: fine-tuned "
                                          f"heavy.* + frozen draft.*).")
                        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)

                        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")
                        
                        # Delete large objects immediately after use to free up memory for the next training epoch
                        del model_state_dict
                        helper.empty_cache()
                        
                    except Exception as e:
                        logger.info_rank0(f"Failed to save HF checkpoint: {e}")

                # Barrier is recommended here to prevent other ranks from starting the next epoch 
                # while Rank 0 is still converting weights, avoiding desync or resource contention
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
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")


    synchronize()
    if args.train.eval_steps:
        _run_eval(global_step)          # final eval
    metrics_logger.close()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # save model in huggingface's format
    if args.train.global_rank == 0 and args.train.save_hf_weights and save_checkpoint_path is not None:
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            output_dir=args.train.output_dir,
            ckpt_manager=args.train.ckpt_manager,
        )
        # HEAVY fine-tune: KEEP the full model (fine-tuned heavy.* + frozen draft.*) -- self-contained DBet ckpt.
        logger.info_rank0(f"HF export: {len(model_state_dict)} tensors (full model: fine-tuned heavy.* + draft.*).")
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
