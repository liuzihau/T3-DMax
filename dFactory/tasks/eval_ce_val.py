# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# Quick CE evaluation: compares pure LLaDA-2.0-mini vs ThinkTalk-LLaDA2 (T3-D) on a
# held-out subset of the training data, at multiple fixed sigma (mask ratio) values.
#
# Held-out set selection:
#   Training shuffles the dataset deterministically using `args.train.seed`. At step
#   15000 of train_steps=174854, training has consumed ~120k samples (=15000 * 8). The
#   last `--val_tail` indices of the same shuffled order are therefore guaranteed unseen
#   at this checkpoint. This script reproduces the same shuffle via torch.randperm with
#   the matching seed and takes that tail as the val set.
#
# Output: a sigma-keyed mean CE table. CE is computed on the noisy-half logits against
# the original tokens, with -100 ignored (prompt + far-tail) -- the same recipe the
# training loop uses, so numbers are directly comparable to wandb's training/loss.
#
# Recommended invocation (run on a second GPU node, ~5-10 min per model):
#
#   # Pure LLaDA baseline:
#   PYTHONPATH=$(pwd)/VeOmni:$(pwd):$PYTHONPATH \
#   python tasks/eval_ce_val.py \
#     --model_path ../LLaDA2.0-mini-moe-merge \
#     --model_type llada \
#     --val_path ./my_data/postprocess_train.jsonl \
#     --val_tail 500 \
#     --output_json ./eval_results_llada.json
#
#   # T3-D at step 15000:
#   PYTHONPATH=$(pwd)/VeOmni:$(pwd):$PYTHONPATH \
#   python tasks/eval_ce_val.py \
#     --model_path ./outputs/<run>/save_checkpoint_path/global_step_15000/hf_ckpt \
#     --model_type t3d \
#     --val_path ./my_data/postprocess_train.jsonl \
#     --val_tail 500 \
#     --output_json ./eval_results_t3d_15000.json

import argparse
import json
import os
import sys
import time
from functools import partial
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "VeOmni")))

# Reuse the training script's mask helpers so eval-time attention pattern is
# byte-identical to training-time.
from tasks.train_t3_dmax_bd_oput import (  # noqa: E402
    block_diffusion_mask,
    talk_self_attn_mask_L,
    talk_cross_attn_mask,
)
from dataset.data_transform import process_mdm_sft_example  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="HF-format checkpoint dir")
    p.add_argument("--model_type", choices=["llada", "t3d"], required=True,
                   help="llada = pure LLaDA-2.0-mini; t3d = ThinkTalk-LLaDA2")
    p.add_argument("--tokenizer_path", default=None,
                   help="default: same as model_path")
    p.add_argument("--val_path", required=True,
                   help="Path to the training jsonl (we use its shuffled tail)")
    p.add_argument("--val_tail", type=int, default=500,
                   help="How many tail samples of the shuffled order to use as val")
    p.add_argument("--seed", type=int, default=42,
                   help="Shuffle seed -- match args.train.seed used in training")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--mask_token_id", type=int, default=156895)
    p.add_argument("--text_keys", default="messages")
    p.add_argument("--sigmas", default="0.25,0.50,0.75,1.00",
                   help="Comma-separated fixed sigma values (one mean CE per value)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_json", default=None,
                   help="Optional path to dump structured results")
    p.add_argument("--limit", type=int, default=None,
                   help="Optional cap on number of val samples actually processed "
                        "(for fast smoke testing of this script)")
    return p.parse_args()


def load_model(args):
    """Load the model based on --model_type."""
    print(f"[eval_ce_val] Loading {args.model_type} from {args.model_path} ...")
    if args.model_type == "llada":
        from models.llada2_moe.modeling_llada2_moe import LLaDA2MoeModelLM
        model = LLaDA2MoeModelLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    elif args.model_type == "t3d":
        from models.think_talk_llada2.modeling_think_talk_llada2 import (
            ThinkTalkLLaDA2ForCausalLM,
        )
        model = ThinkTalkLLaDA2ForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")
    return model


def build_masks(args, injection_mode, device):
    """Pre-build the attention masks needed for forward; one batch_size=1 copy each.

    Returns a dict suitable for use as forward kwargs. T3-D in hybrid_xattn mode
    additionally gets attention_mask_L and cross_attention_mask.
    """
    L = args.max_seq_len
    block_size = args.block_size
    out: Dict[str, torch.Tensor] = {}

    bd_flag = block_diffusion_mask(
        b=None, h=None,
        q_idx=torch.arange(2 * L)[:, None],
        kv_idx=torch.arange(2 * L)[None, :],
        block_size=block_size,
        n=L,
    ).unsqueeze(0).unsqueeze(0)
    bd_mask = torch.zeros_like(bd_flag, dtype=torch.bfloat16)
    bd_mask.masked_fill_(bd_flag.logical_not(), float("-inf"))
    out["attention_mask"] = bd_mask.to(device)

    if injection_mode == "hybrid_xattn":
        self_flag = talk_self_attn_mask_L(
            q_idx=torch.arange(L)[:, None],
            kv_idx=torch.arange(L)[None, :],
            block_size=block_size,
        ).unsqueeze(0).unsqueeze(0)
        self_mask = torch.zeros_like(self_flag, dtype=torch.bfloat16)
        self_mask.masked_fill_(self_flag.logical_not(), float("-inf"))
        out["attention_mask_L"] = self_mask.to(device)

        cross_flag = talk_cross_attn_mask(
            q_idx=torch.arange(L)[:, None],
            kv_idx=torch.arange(2 * L)[None, :],
            block_size=block_size,
            n=L,
        ).unsqueeze(0).unsqueeze(0)
        cross_mask = torch.zeros_like(cross_flag, dtype=torch.bfloat16)
        cross_mask.masked_fill_(cross_flag.logical_not(), float("-inf"))
        out["cross_attention_mask"] = cross_mask.to(device)

    return out


def main():
    args = parse_args()
    device = args.device

    # ---- Load tokenizer + model -------------------------------------------------
    tok_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    model = load_model(args)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[eval_ce_val] model loaded; {n_params/1e9:.2f}B params; on {device}.")

    injection_mode = getattr(model.config, "anchor_injection_mode", "gated_residual")
    print(f"[eval_ce_val] injection_mode={injection_mode}")

    # ---- Load val data via deterministic shuffle of train_path -----------------
    print(f"[eval_ce_val] reading {args.val_path} ...")
    raw_examples: List[Dict] = []
    with open(args.val_path) as f:
        for line in f:
            raw_examples.append(json.loads(line))
    n = len(raw_examples)
    print(f"[eval_ce_val] total examples in file: {n}")

    if n < args.val_tail + 1000:
        print(f"[eval_ce_val] WARNING: file has only {n} samples; val_tail={args.val_tail} "
              f"is close to the full file size. Verify training has not yet reached "
              f"these indices.")

    gen = torch.Generator().manual_seed(args.seed)
    shuffled = torch.randperm(n, generator=gen).tolist()
    val_indices = shuffled[-args.val_tail:]
    if args.limit is not None:
        val_indices = val_indices[: args.limit]
    print(f"[eval_ce_val] using {len(val_indices)} samples (tail of shuffle seed={args.seed}).")

    # ---- Pre-build position_ids and masks -------------------------------------
    L = args.max_seq_len
    noisy_pos = torch.arange(L, dtype=torch.long, device=device)
    clean_pos = torch.arange(L, dtype=torch.long, device=device)
    pos_2L = torch.cat([noisy_pos, clean_pos], dim=0).unsqueeze(0)
    pos_L = noisy_pos.unsqueeze(0)
    cross_pos = torch.cat([noisy_pos, clean_pos], dim=0).unsqueeze(0)

    masks = build_masks(args, injection_mode=injection_mode, device=device)

    # ---- Loop over sigmas -----------------------------------------------------
    sigmas = [float(x) for x in args.sigmas.split(",")]
    results: Dict[float, Dict[str, float]] = {}

    for sigma in sigmas:
        # Fix sigma by collapsing the range -- transform uses uniform-in-range when
        # progress_state is None, so equal low/high gives a deterministic value.
        transform = partial(
            process_mdm_sft_example,
            tokenizer=tokenizer,
            max_seq_len=L,
            text_keys=args.text_keys,
            noise_range=(sigma, sigma),
            mask_token_id=args.mask_token_id,
            progress_state=None,
        )

        loss_sum = 0.0
        position_count = 0
        skipped = 0
        t0 = time.time()
        for k, idx in enumerate(val_indices):
            try:
                transformed = transform(raw_examples[idx])[0]
            except Exception as e:
                skipped += 1
                continue

            noisy_input_ids = transformed["noisy_input_ids"]
            clean_input_ids = transformed["input_ids"]
            labels = transformed["labels"]

            full_ids = torch.cat([noisy_input_ids, clean_input_ids], dim=0).unsqueeze(0).to(device)
            labels_dev = labels.unsqueeze(0).to(device)

            kwargs = {
                "input_ids": full_ids,
                "attention_mask": masks["attention_mask"],
                "position_ids": pos_2L,
                "use_cache": False,
            }
            if args.model_type == "t3d" and injection_mode == "hybrid_xattn":
                kwargs["attention_mask_L"] = masks["attention_mask_L"]
                kwargs["position_ids_L"] = pos_L
                kwargs["cross_attention_mask"] = masks["cross_attention_mask"]
                kwargs["cross_position_ids"] = cross_pos

            with torch.no_grad():
                out = model(**kwargs)
            logits = out.logits
            noisy_logits = logits[:, :L].contiguous()

            loss = F.cross_entropy(
                noisy_logits.view(-1, noisy_logits.shape[-1]),
                labels_dev.view(-1),
                reduction="sum",
                ignore_index=-100,
            )
            valid = int((labels_dev != -100).sum().item())
            loss_sum += float(loss.item())
            position_count += valid

            if (k + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (k + 1) / max(elapsed, 1e-6)
                eta = (len(val_indices) - (k + 1)) / max(rate, 1e-6)
                print(f"  sigma={sigma:.2f}  [{k+1}/{len(val_indices)}]  "
                      f"running_CE={loss_sum / max(position_count, 1):.4f}  "
                      f"{rate:.1f} ex/s  ETA {eta:.0f}s")

        mean_ce = loss_sum / max(position_count, 1)
        results[sigma] = {
            "mean_ce": mean_ce,
            "n_samples": len(val_indices) - skipped,
            "n_positions": position_count,
            "skipped": skipped,
        }
        print(f"[eval_ce_val] sigma={sigma:.2f}  mean_CE={mean_ce:.4f}  "
              f"(over {position_count} positions, {len(val_indices)-skipped} samples, "
              f"{skipped} skipped)")

    # ---- Report -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Model: {args.model_path}  (type={args.model_type})")
    print(f"Val: tail {args.val_tail} of shuffle seed={args.seed} from {args.val_path}")
    print("=" * 60)
    print(f"{'sigma':>8}  {'mean_CE':>10}  {'n_samples':>10}  {'n_positions':>12}")
    for sigma in sigmas:
        r = results[sigma]
        print(f"{sigma:>8.2f}  {r['mean_ce']:>10.4f}  {r['n_samples']:>10d}  {r['n_positions']:>12d}")

    if args.output_json:
        payload = {
            "model_path": args.model_path,
            "model_type": args.model_type,
            "val_path": args.val_path,
            "val_tail": args.val_tail,
            "seed": args.seed,
            "max_seq_len": args.max_seq_len,
            "block_size": args.block_size,
            "results": {str(s): results[s] for s in sigmas},
            "n_params_billion": n_params / 1e9,
            "injection_mode": injection_mode,
        }
        with open(args.output_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[eval_ce_val] Saved structured results to {args.output_json}")


if __name__ == "__main__":
    main()
