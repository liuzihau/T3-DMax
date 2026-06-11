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
ModelRegistry.register_modeling_path("models.llada2_moe")
from dataset.data_transform import process_mdm_tokenized_example, process_mdm_sft_example
from dataset import build_local_dataset
import random

# T3-D top-K talk: anchor-free think/talk. The frozen full think provides per-position
# top-K candidates; the talk (this trainer's `model`, = merged_10L) consumes them as the
# input embedding at masked positions. See probe_runner/T3D_TOPK_TALK_INTEGRATION.md.
from tasks.t3d_topk_talk import (build_talk_inputs_embeds, load_causal_lm, set_talk_trainable,
                                 confident_prefix_commit, think_distill_loss,
                                 build_think_next_teacher)


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
    # T3-D top-K talk ----------------------------------------------------------------
    think_path: str = field(
        default=None,
        metadata={"help": "Path to the FROZEN full think model (e.g. DMax-Math-16B). Loaded "
                          "replicated per rank (no FSDP, no grad); supplies the top-K candidates."}
    )
    t3_top_k: int = field(
        default=10,
        metadata={"help": "K for think's top-K candidate soft-embedding fed to the talk."}
    )
    t3_train_layers: str = field(
        default="6,8",
        metadata={"help": "Talk stack indices to train (merged-only). For keep=0-5,12,19 the "
                          "two merged-representative layers are at positions 6 and 8."}
    )
    t3_mask_token_id: int = field(
        default=156895,
        metadata={"help": "LLaDA-2.0-mini's [MASK] token id."}
    )
    t3_val_every: int = field(
        default=0,
        metadata={"help": "Run quick anchor-free top-K val (CE + token-match acc) every N "
                          "steps (0=off). Val also runs at every save_steps for ckpt retention."}
    )
    t3_val_size: int = field(
        default=64,
        metadata={"help": "Number of val examples (tail of the train jsonl)."}
    )
    t3_metrics_jsonl: str = field(
        default="",
        metadata={"help": "Local metrics jsonl for self-plotting (default: <output_dir>/metrics.jsonl)."}
    )
    t3_keep_best_n: int = field(
        default=3,
        metadata={"help": "Keep only the N best hf checkpoints by val CE (0=keep all)."}
    )
    t3_rollout_commit_frac: float = field(
        default=1.0,
        metadata={"help": "On-policy rollout gate (flag=True): >0 enables the rollout (commit the "
                          "talk's own argmax before the grad forward, so it trains on its own "
                          "commits — fixes exposure-bias collapse); 0 = pure teacher-forced. "
                          "Commit SELECTION is the confidence-prefix rule (t3_commit_threshold), "
                          "matching inference (not a random fraction)."}
    )
    t3_commit_threshold: float = field(
        default=0.3,
        metadata={"help": "Confidence threshold for the rollout's per-block left-to-right prefix "
                          "commit (DMax decode_uniform tau; matches inference)."}
    )
    t3_curriculum_step: float = field(
        default=0.0,
        metadata={"help": "Mask-ratio CURRICULUM enable + step. 0 = off (sigma uniform in [noise_range]). "
                          ">0 enables a multi-step ramp from noise_range_low to high: each time the talk "
                          "BEATS think's top-1 (acc_tf > acc_think, i.e. positive gain) AND acc_tf >= "
                          "t3_full_mask_after_acc, the shared sigma-progress advances by this step. "
                          "0.5 -> low / mid / high (e.g. reveal 25% -> 12.5% -> 0%); 1.0 -> low / high. "
                          "Requires noise_range_low != noise_range_high."}
    )
    t3_full_mask_after_acc: float = field(
        default=0.0,
        metadata={"help": "Curriculum acc FLOOR: don't advance the mask-ratio step until val/acc_tf >= "
                          "this (prevents advancing on early noise)."}
    )
    t3_curriculum_margin: float = field(
        default=0.03,
        metadata={"help": "Curriculum advance gate: advance when the talk has CAUGHT UP to think's "
                          "top-1, i.e. acc_tf >= acc_think - margin. (The talk MATCHES, never beats, "
                          "think — that's its ceiling — so a strict 'beats think' gate would stall "
                          "forever. This fires once the talk has learned all it can at the level.)"}
    )
    t3_keep_mask_residual: bool = field(
        default=False,
        metadata={"help": "Path-B top-K input variant. False = no-mask renorm-in-top-K (clean candidate "
                          "signal; use for the Stage-1 cold start). True = top-K + mask-residual + "
                          "renormalize = the EXACT inference input -> use for Stage-2 on-policy so "
                          "the rollout's commits and the trained input match decode time."}
    )
    # Path-A think->talk distillation (Stage-1 cold start) --------------------------
    t3_distill_beta: float = field(
        default=0.0,
        metadata={"help": "Path-A (mask path) think-distillation weight. 0 = OFF (legacy: pure gold "
                          "CE, think unused on Path A). >0 = forward additionally the FROZEN think on "
                          "the masked input and add beta * forward-KL(think||talk) at the predict "
                          "positions, so the 16B think's dark knowledge (full top-K) trains the talk, "
                          "not just one-hot gold. Mass-covering -> cold-start alignment to parity."}
    )
    t3_distill_alpha: float = field(
        default=1.0,
        metadata={"help": "Path-A gold-CE weight in the hybrid loss alpha*CE(gold)+beta*KL(think||talk). "
                          "Keep the gold term so think's errors are corrected and talk retains a path "
                          "to exceed think (set 0 for pure-distillation ablation). Only used when "
                          "t3_distill_beta>0; Path B is unaffected (pure gold CE)."}
    )
    t3_distill_temp: float = field(
        default=1.0,
        metadata={"help": "Distillation softmax temperature (both think and talk sides); loss scaled "
                          "by T^2. 1.0 = no softening; 2.0 = classic Hinton softening (surfaces think's "
                          "rank 2..K more strongly). Only used when t3_distill_beta>0."}
    )
    # Path-B think_{s+2} distillation (Stage-1 cold start) --------------------------
    t3_pathb_kl_gamma: float = field(
        default=0.0,
        metadata={"help": "Path-B (top-K path) s+2 distillation weight. 0 = OFF (legacy: pure gold "
                          "CE). >0 = DMax-OPUT-style: hard-commit think's s+1 confident prefix into "
                          "the talk's input (Portion 1, kept in the loss), advance think ONE DMax "
                          "decode step to think_{s+2}, and add gamma * forward-KL(think_{s+2}||talk) "
                          "to the gold CE. think-sourced commits = off-policy (stage-1); stage-2 "
                          "swaps to talk's own commits. Costs a SECOND 16B think forward on Path B."}
    )
    t3_pathb_kl_temp: float = field(
        default=1.0,
        metadata={"help": "Path-B s+2 distillation softmax temperature (loss scaled by T^2). Only "
                          "used when t3_pathb_kl_gamma>0. Keep gamma modest so gold (truth) wins the "
                          "flip decision at stuck-wrong forking positions (KL is a dope, not driver)."}
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


# ============================================================================
# T3-D top-K: quick validation + local metric logging + best-N ckpt retention
# ============================================================================
def _append_jsonl(path, record):
    """Append one metric row to a local jsonl (for self-plotting). Rank-0 only."""
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning_rank0(f"[t3d metrics] write failed {path}: {e}")


@torch.no_grad()
def _run_topk_val(think, talk, embedding, mask_id, val_examples, transform, attn_proto, device,
                  top_k, rollout_frac=0.5, keep_mask_residual=False, do_rollout=True):
    """Quick anchor-free top-K val. Reports TWO sets of CE + token-match acc on the masked
    predict positions:
      * `_tf`   : teacher-forced -- clean context (the base rerank ability; looks easy).
      * `_roll` : rollout -- a `rollout_frac` slice of masked positions is committed with the
                  talk's OWN argmax (own imperfect commits as context), the rest re-predicted.
                  This mirrors the on-policy/inference condition and is the metric that should
                  track GSM8K. The talk's first (TF) forward IS the rollout's first pass, so
                  this costs 2 talk forwards/example total."""
    was_training = talk.training
    talk.eval()
    th_ce = th_ok = th_n = 0          # think's own top-1 (the candidate source) — does the talk beat it?
    tf_ce = tf_ok = tf_n = 0
    mk_ce = mk_ok = mk_n = 0
    rl_ce = rl_ok = rl_n = 0
    attn = attn_proto.to(device)
    for ex in val_examples:
        try:
            t = transform(ex)
            t = t[0] if isinstance(t, (list, tuple)) else t
        except Exception:
            continue
        noisy, clean, labels = t["noisy_input_ids"], t["input_ids"], t["labels"]
        L = noisy.shape[0]
        full = torch.cat([noisy, clean], 0).unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        pos = torch.cat([torch.arange(L), torch.arange(L)], 0).unsqueeze(0).to(device)
        think_logits = think(inputs_embeds=embedding(full), attention_mask=attn,
                             position_ids=pos, use_cache=False, return_dict=True).logits
        valid = labels != -100
        if int(valid.sum()) == 0:
            continue
        # --- think's own top-1 (the candidate source; reuse think_logits, no extra forward) ---
        th_logits = think_logits[:, :L]
        th_ce += F.cross_entropy(th_logits.reshape(-1, th_logits.shape[-1]).float(),
                                 labels.reshape(-1), ignore_index=-100, reduction="sum").item()
        th_ok += int(((th_logits.argmax(-1) == labels) & valid).sum())
        th_n += int(valid.sum())
        # --- teacher-forced (clean context) ---
        tf_embeds = build_talk_inputs_embeds(full, think_logits, embedding, mask_id,
                                             mode="topk_soft", top_k=top_k, keep_mask_residual=keep_mask_residual)
        tf_logits = talk(inputs_embeds=tf_embeds, attention_mask=attn, position_ids=pos,
                         use_cache=False, return_dict=True).logits[:, :L]
        tf_ce += F.cross_entropy(tf_logits.reshape(-1, tf_logits.shape[-1]).float(),
                                 labels.reshape(-1), ignore_index=-100, reduction="sum").item()
        tf_ok += int(((tf_logits.argmax(-1) == labels) & valid).sum())
        tf_n += int(valid.sum())
        # --- PURE MASK path (no think) — the baseline: does top-K beat [MASK]? ---
        mk_embeds = build_talk_inputs_embeds(full, None, embedding, mask_id, mode="mask")
        mk_logits = talk(inputs_embeds=mk_embeds, attention_mask=attn, position_ids=pos,
                         use_cache=False, return_dict=True).logits[:, :L]
        mk_ce += F.cross_entropy(mk_logits.reshape(-1, mk_logits.shape[-1]).float(),
                                 labels.reshape(-1), ignore_index=-100, reduction="sum").item()
        mk_ok += int(((mk_logits.argmax(-1) == labels) & valid).sum())
        mk_n += int(valid.sum())
        # --- rollout (own commits as context) — ONLY when on-policy is active (Stage 2) ---
        if do_rollout:
            argmax_tf = tf_logits.argmax(-1)                               # talk's own preds
            committed = valid & (torch.rand_like(argmax_tf.float()) < rollout_frac)
            still = valid & (~committed)
            if int(still.sum()) > 0:
                full_roll = full.clone()
                full_roll[:, :L] = torch.where(committed, argmax_tf, full_roll[:, :L])
                rl_embeds = build_talk_inputs_embeds(full_roll, think_logits, embedding, mask_id,
                                                     mode="topk_soft", top_k=top_k, keep_mask_residual=keep_mask_residual)
                rl_logits = talk(inputs_embeds=rl_embeds, attention_mask=attn, position_ids=pos,
                                 use_cache=False, return_dict=True).logits[:, :L]
                labels_still = labels.clone()
                labels_still[~still] = -100
                rl_ce += F.cross_entropy(rl_logits.reshape(-1, rl_logits.shape[-1]).float(),
                                         labels_still.reshape(-1), ignore_index=-100, reduction="sum").item()
                rl_ok += int(((rl_logits.argmax(-1) == labels) & still).sum())
                rl_n += int(still.sum())
    if was_training:
        talk.train()
    out = {"val/ce_think": (th_ce / th_n if th_n else float("nan")),
           "val/acc_think": (th_ok / th_n if th_n else float("nan")),
           "val/ce_tf": (tf_ce / tf_n if tf_n else float("nan")),
           "val/acc_tf": (tf_ok / tf_n if tf_n else float("nan")),
           "val/ce_mask": (mk_ce / mk_n if mk_n else float("nan")),
           "val/acc_mask": (mk_ok / mk_n if mk_n else float("nan")),
           "val/ce_roll": (rl_ce / rl_n if rl_n else float("nan")),
           "val/acc_roll": (rl_ok / rl_n if rl_n else float("nan"))}
    return out


def _prune_best_ckpts(saved, keep_n):
    """Keep the LATEST keep_n step dirs; rmtree the rest. (Keep-latest, not best-ce: under a
    mask-ratio curriculum ce RISES with sigma, so 'best ce' would prune the most-trained ckpt —
    exactly the one we chain from. Storage-friendly: only keep_n dirs ever on disk.)"""
    import shutil
    if keep_n <= 0 or len(saved) <= keep_n:
        return saved
    ordered = sorted(saved, key=lambda d: d["step"], reverse=True)     # latest first
    keep, drop = ordered[:keep_n], ordered[keep_n:]
    for d in drop:
        shutil.rmtree(d["dir"], ignore_errors=True)
        logger.info_rank0(f"[t3d ckpt] pruned {d['dir']} (step {d['step']}); keeping latest {keep_n}")
    return keep


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
    noise_progress = None     # mask-ratio curriculum handle (set in the conversation branch)
    if args.data.data_type == "conversation":
        if not tokenizer.chat_template:
            raise ValueError(f"No chat template found in the tokenizer.")

        # T3-D mask-ratio curriculum: when t3_full_mask_after_acc>0, sigma is gated by a shared
        # progress value (0 -> noise_range_low, 1 -> noise_range_high). The val loop flips it to
        # 1.0 once val/acc_tf crosses the threshold (sigma low -> high = e.g. 75% -> 100% mask).
        noise_progress = (mp.Value("d", 0.0)
                          if args.train.t3_curriculum_step > 0
                          and args.data.noise_range_low != args.data.noise_range_high else None)
        transform = partial(
            process_mdm_sft_example,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            noise_range=(args.data.noise_range_low, args.data.noise_range_high),
            mask_token_id=156895,
            progress_state=noise_progress,
        )
    elif args.data.data_type == "tokenid":
        transform = partial(
            process_mdm_tokenized_example,
            max_seq_len=args.data.max_seq_len,
            text_keys=args.data.text_keys,
            noise_range=(args.data.noise_range_low, args.data.noise_range_high),
            mask_token_id=156895, 
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
    # T3-D: load the talk with the SAME proven loader as think. build_foundation_model
    # random-inits merged_10L's MoE experts (key-format mismatch vs merge_layers --save_hf;
    # the giant "...initialize them" list = lost merge init). load_causal_lm loads them
    # fully (verified: clean load in the smoke + for think here).
    model = load_causal_lm(args.model.model_path, get_device_type(),
                           dtype=torch.bfloat16, attn_implementation=args.model.attn_implementation)
    model_config = model.config
    helper.print_device_mem_info("VRAM usage after building model")

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

    # T3-D top-K: frozen full think (replicated per rank, no FSDP, no grad) + merged-only
    # trainable scope on the talk (= this `model`). Must run BEFORE build_optimizer so the
    # optimizer only sees the trainable merged layers.
    think = load_causal_lm(args.train.think_path, get_device_type(), dtype=torch.bfloat16)
    for p in think.parameters():
        p.requires_grad_(False)
    think_emb = think.get_input_embeddings()
    _train_idx = [int(x) for x in args.train.t3_train_layers.split(",") if x != ""]
    _n_tr = set_talk_trainable(model, _train_idx, freeze_embed_head=True)
    logger.info_rank0(f"[T3-D top-K] frozen think loaded; trainable talk layers "
                      f"{_train_idx} -> {_n_tr:,} params (merged-only)")

    # T3-D top-K: val set (tail of the train jsonl) + local metrics + best-N ckpt tracking.
    _val_examples = []
    if args.train.t3_val_every > 0 or args.train.t3_keep_best_n > 0:
        try:
            with open(args.data.train_path, "r", encoding="utf-8") as _fh:
                _tail = _fh.readlines()[-args.train.t3_val_size:]
            _val_examples = [json.loads(s) for s in (l.strip() for l in _tail) if s]
            logger.info_rank0(f"[t3d val] {len(_val_examples)} val examples (tail of train jsonl)")
        except Exception as _e:
            logger.warning_rank0(f"[t3d val] could not load val set: {_e}; val/retention disabled")
    _metrics_path = args.train.t3_metrics_jsonl or os.path.join(args.train.output_dir, "metrics.jsonl")
    _saved_ckpts = []          # [{val_ce, dir, step}] for best-N retention
    _last_val_ce = float("nan")

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

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, epochs: {args.train.num_train_epochs}"
    )
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

                labels = micro_batch.pop("labels", None)


                #=============== T3-D two-path top-K injection (+ on-policy rollout) ==============
                # TWO PATHS (selected by `flag`, ~50/50 from the data), matching DMax's training
                # (the talk inherits DMax's merged weights, so keep both):
                #   Path A (flag=False): masked -> [MASK]  (standard MDM; think NOT needed)
                #   Path B (flag=True):  masked -> think's top-K soft-embed (the inference input)
                # flag=True additionally runs an ON-POLICY rollout (if t3_rollout_commit_frac>0):
                # the talk predicts, the CONFIDENCE-PREFIX of masked positions is committed with
                # the talk's OWN argmax, and the grad forward then sees its own commits as context
                # (fixes exposure-bias collapse; mirrors inference).
                mask_token_id = args.train.t3_mask_token_id
                flag = bool(micro_batch["flag"].item())
                mode = "topk_soft" if flag else "mask"
                # Path B always needs think (top-K input). Path A forwards think ONLY to
                # distill (t3_distill_beta>0): the loss matches think's distribution; the
                # Path-A input stays bare [MASK] (build_talk_inputs_embeds mode='mask'
                # ignores think_logits), so think is a teacher, not an input, here.
                _distill_A = (not flag) and args.train.t3_distill_beta > 0
                think_logits = None
                if flag or _distill_A:
                    with torch.no_grad():
                        think_logits = think(
                            inputs_embeds=think_emb(micro_batch["input_ids"]),
                            attention_mask=micro_batch["attention_mask"],
                            position_ids=micro_batch["position_ids"],
                            use_cache=False,
                        ).logits
                        if flag and args.train.t3_rollout_commit_frac > 0:
                            L0 = noisy_input_ids.shape[1]
                            # commit-deciding forward uses the EXACT inference input (mask-residual
                            # when t3_keep_mask_residual), so the commits match decode time.
                            _roll_in = build_talk_inputs_embeds(
                                micro_batch["input_ids"], think_logits, think_emb, mask_token_id,
                                mode="topk_soft", top_k=args.train.t3_top_k,
                                keep_mask_residual=args.train.t3_keep_mask_residual)
                            _roll_logits = model(
                                inputs_embeds=_roll_in, attention_mask=micro_batch["attention_mask"],
                                position_ids=micro_batch["position_ids"], use_cache=False,
                                output_router_logits=False).logits[:, :L0]
                            _masked = micro_batch["input_ids"][:, :L0] == mask_token_id
                            # CONFIDENCE-PREFIX commit (inference rule), not a random fraction.
                            _argmax, _commit = confident_prefix_commit(
                                _roll_logits, _masked, args.train.block_size, args.train.t3_commit_threshold)
                            micro_batch["input_ids"][:, :L0] = torch.where(
                                _commit, _argmax, micro_batch["input_ids"][:, :L0])

                # ---- Path-B (stage-1) think_{s+2} teacher + Portion-1 hard-commit ----------------
                # OPUT-style, think-sourced (off-policy): advance think ONE DMax step to think_{s+2}
                # (the KL teacher), and hard-commit that SAME s+1 confident prefix into the talk's
                # input as Portion 1. Labels untouched -> Portion 1 stays gold (in CE + KL); the
                # curriculum reveal stays -100 (no loss). Build the teacher from the ORIGINAL s+1
                # state, THEN write Portion 1 (one decode decision drives both). Stage-2 will swap
                # the commit source to the talk's own argmax (on-policy).
                think_s2_logits = None
                if flag and args.train.t3_pathb_kl_gamma > 0:
                    with torch.no_grad():
                        L0b = noisy_input_ids.shape[1]
                        think_s2_logits, _b_argmax, _b_commit = build_think_next_teacher(
                            think, think_emb, micro_batch["input_ids"], think_logits, L0b,
                            mask_token_id, block_size=args.train.block_size,
                            threshold=args.train.t3_commit_threshold, top_k=args.train.t3_top_k,
                            attention_mask=micro_batch["attention_mask"],
                            position_ids=micro_batch["position_ids"])
                        micro_batch["input_ids"][:, :L0b] = torch.where(
                            _b_commit, _b_argmax, micro_batch["input_ids"][:, :L0b])
                talk_embeds = build_talk_inputs_embeds(
                    micro_batch["input_ids"], think_logits, think_emb, mask_token_id,
                    mode=mode, top_k=args.train.t3_top_k,
                    keep_mask_residual=args.train.t3_keep_mask_residual,
                )
                #======================================================================================


                with model_fwd_context:
                    # T3-D: talk forward on the top-K-injected embeds (anchor-free).
                    logits: "torch.Tensor" = model(
                        inputs_embeds=talk_embeds,
                        attention_mask=micro_batch["attention_mask"],
                        position_ids=micro_batch["position_ids"],
                        use_cache=False, output_router_logits=False,
                    ).logits
                    if args.train.block_diffusion_mode:
                        noisy_logits = logits[:, :noisy_input_ids.shape[1]].contiguous()
                    else:
                        noisy_logits = logits

                    if args.train.same_token_labels:
                        unscaled_loss = torch.nn.functional.cross_entropy(
                            noisy_logits.view(-1, noisy_logits.shape[-1]),
                            labels.view(-1),
                            reduction="none",
                        )
                        ce_mean = unscaled_loss.sum() / (labels != -100).sum()
                    else:
                        shifted_noisy_logits = noisy_logits[:, :-1, :].contiguous()
                        shifted_labels = labels[:, 1:].contiguous()
                        unscaled_loss = torch.nn.functional.cross_entropy(
                            shifted_noisy_logits.view(-1, shifted_noisy_logits.shape[-1]),
                            shifted_labels.view(-1),
                            reduction="none",
                        ).view(shifted_noisy_logits.shape[0], -1)
                        ce_mean = unscaled_loss.sum() / (shifted_labels != -100).sum()

                    # Path-A think->talk distillation: hybrid alpha*CE(gold) + beta*KL(think||talk)
                    # at the predict positions. Path B (flag) and distill-off keep pure gold CE.
                    if _distill_A:
                        L_n = noisy_input_ids.shape[1]
                        if args.train.same_token_labels:
                            kl = think_distill_loss(
                                noisy_logits, think_logits[:, :L_n], labels,
                                temperature=args.train.t3_distill_temp)
                        else:
                            kl = think_distill_loss(
                                shifted_noisy_logits, think_logits[:, :L_n][:, 1:].contiguous(),
                                shifted_labels, temperature=args.train.t3_distill_temp)
                        loss = (args.train.t3_distill_alpha * ce_mean
                                + args.train.t3_distill_beta * kl) / len(micro_batches)
                    elif think_s2_logits is not None:
                        # Path-B s+2 distillation: gold CE + gamma * forward-KL(think_{s+2}||talk),
                        # both over the predict positions (Portion 1 + still-masked; reveal is -100).
                        if args.train.same_token_labels:
                            kl_b = think_distill_loss(noisy_logits, think_s2_logits, labels,
                                                      temperature=args.train.t3_pathb_kl_temp)
                        else:
                            kl_b = think_distill_loss(shifted_noisy_logits,
                                                      think_s2_logits[:, 1:].contiguous(), shifted_labels,
                                                      temperature=args.train.t3_pathb_kl_temp)
                        loss = (ce_mean + args.train.t3_pathb_kl_gamma * kl_b) / len(micro_batches)
                    else:
                        loss = ce_mean / len(micro_batches)

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                del micro_batch

            # Prefer model-provided clip_grad_norm_ (now both FSDP1 and FSDP2 registers custom grad norm clipping)
            if hasattr(model, "clip_grad_norm_"):
                _gn = model.clip_grad_norm_(args.train.max_grad_norm)
                grad_norm = _gn.item() if hasattr(_gn, "item") else float(_gn)
            else:
                # T3-D: DDP/single-GPU has no registered clip_grad_norm_; PyTorch default is
                # correct here (silenced the per-step info log).
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm)

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

                # T3-D: local metrics row (+ quick val at val_every / save_steps) for self-plotting.
                _rec = {"step": global_step, "train_loss": total_loss, "grad_norm": grad_norm, "lr": lr}
                _do_val = bool(_val_examples) and (
                    (args.train.t3_val_every and global_step % args.train.t3_val_every == 0)
                    or (args.train.save_steps and global_step % args.train.save_steps == 0))
                if _do_val:
                    _vm = _run_topk_val(think, model, think_emb, args.train.t3_mask_token_id,
                                        _val_examples, transform, block_diffusion_attn_mask_prototype,
                                        get_device_type(), args.train.t3_top_k,
                                        keep_mask_residual=args.train.t3_keep_mask_residual,
                                        do_rollout=args.train.t3_rollout_commit_frac > 0)
                    _rec.update(_vm)
                    # retention by the inference-relevant rollout CE (fall back to TF if nan)
                    _last_val_ce = _vm["val/ce_roll"]
                    if _last_val_ce != _last_val_ce:  # nan
                        _last_val_ce = _vm["val/ce_tf"]
                    if args.train.use_wandb:
                        wandb.log(_vm, step=global_step)
                    _roll_str = (f"acc_roll={_vm['val/acc_roll']:.4f} "
                                 if _vm["val/acc_roll"] == _vm["val/acc_roll"] else "")  # nan => off (Stage 1)
                    logger.info_rank0(f"[t3d val] step {global_step} "
                                      f"acc_think={_vm['val/acc_think']:.4f}(think top1) "
                                      f"acc_tf={_vm['val/acc_tf']:.4f}(talk topK) "
                                      f"acc_mask={_vm['val/acc_mask']:.4f}(mask) {_roll_str}| "
                                      f"[topK gain={_vm['val/acc_tf']-_vm['val/acc_mask']:+.3f}] "
                                      f"[talk-vs-think={_vm['val/acc_tf']-_vm['val/acc_think']:+.3f}] "
                                      f"ce: think={_vm['val/ce_think']:.3f} talk={_vm['val/ce_tf']:.3f}")
                    # mask-ratio curriculum: advance one step (reveal ratio down -> sigma up) when the
                    # talk has CAUGHT UP to think's top-1 (acc_tf within margin of acc_think = its ceiling)
                    # AND clears the acc floor. One-way. (A strict 'beats think' gate never fires since the
                    # talk matches, not beats, think.)
                    _caught_up = (_vm["val/acc_tf"] == _vm["val/acc_tf"]              # not nan
                                  and _vm["val/acc_tf"] >= _vm["val/acc_think"] - args.train.t3_curriculum_margin)
                    if (noise_progress is not None and noise_progress.value < 1.0
                            and _caught_up
                            and _vm["val/acc_tf"] >= args.train.t3_full_mask_after_acc):
                        noise_progress.value = min(1.0, noise_progress.value + args.train.t3_curriculum_step)
                        _sigma = (args.data.noise_range_low + noise_progress.value
                                  * (args.data.noise_range_high - args.data.noise_range_low))
                        _rec["curriculum"] = f"sigma={_sigma:.3f}"
                        logger.info_rank0(f"[t3d curriculum] step {global_step} talk caught up to think "
                                          f"(acc_tf {_vm['val/acc_tf']:.3f} vs acc_think "
                                          f"{_vm['val/acc_think']:.3f}, margin {args.train.t3_curriculum_margin}) "
                                          f"-> progress {noise_progress.value:.2f}, sigma~{_sigma:.3f} "
                                          f"(reveal~{(1-_sigma)*100:.1f}%)")
                _append_jsonl(_metrics_path, _rec)

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
                        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
                        
                        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")
                        
                        # Delete large objects immediately after use to free up memory for the next training epoch
                        del model_state_dict
                        helper.empty_cache()

                        # T3-D: best-N retention. _last_val_ce was refreshed by the val that
                        # runs at every save_steps above. Keep the N lowest-CE step dirs.
                        if args.train.t3_keep_best_n > 0:
                            _saved_ckpts.append({"val_ce": _last_val_ce, "dir": save_checkpoint_path,
                                                 "step": global_step})
                            _saved_ckpts = _prune_best_ckpts(_saved_ckpts, args.train.t3_keep_best_n)
                            _append_jsonl(_metrics_path, {"step": global_step, "event": "save",
                                                          "val_ce": _last_val_ce, "dir": save_checkpoint_path,
                                                          "kept": [d["step"] for d in _saved_ckpts]})

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
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
