# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# Block-diffusion GSM8K generation for T3-D (and pure LLaDA-2.0-mini as baseline).
#
# Decoding protocol (mirrors DMax's `generate_with_prefix_cache` in
# dInfer/python/dinfer/decoding/generate_fastdllm.py, adapted to our doubled-sequence
# T3-D forward):
#   1. Build prompt via the model's chat template.
#   2. Construct a length-L working sequence: [prompt, [MASK]*gen_length, pad].
#   3. For each contiguous gen_length / block_size block of response tokens:
#      For up to `max_iter_per_block` iterations:
#        a. Run model.forward on the doubled input [working, working]. (Doubling matches
#           training; clean half is set to a mirror of the noisy half -- inference
#           doesn't have an original "clean" half. This is approximate, but cheap and
#           the alternative is re-deriving inference architecture from scratch.)
#        b. Take logits[:, :L] (noisy-half predictions).
#        c. Among masked positions in the current block, find those with softmax peak
#           > threshold (DMax's reveal rule). Commit them.
#        d. Fallback: if no position passes threshold, commit the highest-confidence
#           masked position (guaranteed progress).
#        e. If all positions in the block are committed, advance to next block.
#   4. Decode the response area to text.
#
# Output: jsonl where each row is `{question, answer, ground_truth}`. Grade with
# `tasks/gsm8k_grade.py` (vendored from DMax) to get accuracy.
#
# Usage:
#   # T3-D from an HF checkpoint:
#   PYTHONPATH=$(pwd)/VeOmni:$(pwd):$PYTHONPATH \
#   python tasks/gsm8k_generate.py \
#     --model_path ./outputs/<run>/checkpoints/global_step_<N>/hf_ckpt \
#     --tokenizer_path ../LLaDA2.0-mini-moe-merge \
#     --model_type t3d \
#     --output_path ./outputs/gsm8k_t3d_step<N>.jsonl
#
#   # Pure LLaDA baseline:
#   PYTHONPATH=$(pwd)/VeOmni:$(pwd):$PYTHONPATH \
#   python tasks/gsm8k_generate.py \
#     --model_path ../LLaDA2.0-mini-moe-merge \
#     --model_type llada \
#     --output_path ./outputs/gsm8k_llada_baseline.jsonl
#
# Then:
#   python tasks/gsm8k_grade.py --pred-path ./outputs/gsm8k_t3d_step<N>.jsonl

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "VeOmni")))

# Reuse the same mask builders the training and CE-eval scripts use, so attention
# pattern is byte-identical to training-time.
from tasks.train_t3_dmax_bd_oput import (  # noqa: E402
    block_diffusion_mask,
    talk_self_attn_mask_L,
    talk_cross_attn_mask,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="HF checkpoint dir")
    p.add_argument("--model_type", required=True, choices=["llada", "t3d"])
    p.add_argument("--tokenizer_path", default=None,
                   help="default: same as model_path. Set to LLaDA's dir if model dir has no tokenizer files.")
    p.add_argument("--output_path", required=True,
                   help="Path to write predictions jsonl")
    p.add_argument("--dataset_name", default="openai/gsm8k")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N examples (for quick tests)")
    # Generation knobs
    p.add_argument("--max_seq_len", type=int, default=2048,
                   help="L (single half of doubled sequence)")
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--gen_length", type=int, default=512,
                   help="Number of response tokens to decode")
    p.add_argument("--max_iter_per_block", type=int, default=8,
                   help="Max talk iterations within each block before forced advance")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Softmax-peak threshold for committing a position (DMax default).")
    p.add_argument("--mask_token_id", type=int, default=156895)
    p.add_argument("--pad_token_id", type=int, default=156892)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_model(args):
    """Load model with fused-MoE dispatch (mirrors eval_ce_val.py)."""
    if args.model_type == "llada":
        from models.llada2_moe.configuration_llada2_moe import LLaDA2MoeConfig
        from models.llada2_moe.modeling_llada2_moe import LLaDA2MoeModelLM

        config = LLaDA2MoeConfig.from_pretrained(args.model_path)
        if not config.model_type.endswith("_veomni"):
            config.model_type = config.model_type + "_veomni"
        if getattr(config, "moe_implementation", None) != "fused":
            config.moe_implementation = "fused"
        model = LLaDA2MoeModelLM.from_pretrained(
            args.model_path, config=config,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        )
    elif args.model_type == "t3d":
        from models.think_talk_llada2.configuration_think_talk_llada2 import (
            ThinkTalkLLaDA2Config,
        )
        from models.think_talk_llada2.modeling_think_talk_llada2 import (
            ThinkTalkLLaDA2ForCausalLM,
        )

        config = ThinkTalkLLaDA2Config.from_pretrained(args.model_path)
        if not config.model_type.endswith("_veomni"):
            config.model_type = config.model_type + "_veomni"
        if getattr(config, "moe_implementation", None) != "fused":
            config.moe_implementation = "fused"
        model = ThinkTalkLLaDA2ForCausalLM.from_pretrained(
            args.model_path, config=config,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        )
    return model


def build_attention_masks(L: int, block_size: int, device: str, model_type: str):
    """Pre-build all attention masks (batch=1) and position_ids needed for the forward.

    LLaDA only needs the 2L block-diffusion mask. T3-D in hybrid_xattn mode also needs
    the L self-attn mask and the L-by-2L cross-attn mask.
    """
    dtype = torch.bfloat16

    bd_flag = block_diffusion_mask(
        b=None, h=None,
        q_idx=torch.arange(2 * L)[:, None],
        kv_idx=torch.arange(2 * L)[None, :],
        block_size=block_size,
        n=L,
    ).unsqueeze(0).unsqueeze(0)
    attn_mask_2L = torch.zeros_like(bd_flag, dtype=dtype)
    attn_mask_2L.masked_fill_(bd_flag.logical_not(), float("-inf"))
    attn_mask_2L = attn_mask_2L.to(device)

    noisy_pos = torch.arange(L, dtype=torch.long, device=device)
    clean_pos = torch.arange(L, dtype=torch.long, device=device)
    pos_2L = torch.cat([noisy_pos, clean_pos], dim=0).unsqueeze(0)

    masks = {
        "attention_mask": attn_mask_2L,
        "position_ids": pos_2L,
    }

    if model_type == "t3d":
        # L self-attn (talk) and L x 2L cross-attn (talk).
        self_flag = talk_self_attn_mask_L(
            q_idx=torch.arange(L)[:, None],
            kv_idx=torch.arange(L)[None, :],
            block_size=block_size,
        ).unsqueeze(0).unsqueeze(0)
        attn_mask_L = torch.zeros_like(self_flag, dtype=dtype)
        attn_mask_L.masked_fill_(self_flag.logical_not(), float("-inf"))
        masks["attention_mask_L"] = attn_mask_L.to(device)
        masks["position_ids_L"] = noisy_pos.unsqueeze(0)

        cross_flag = talk_cross_attn_mask(
            q_idx=torch.arange(L)[:, None],
            kv_idx=torch.arange(2 * L)[None, :],
            block_size=block_size,
            n=L,
        ).unsqueeze(0).unsqueeze(0)
        cross_attn_mask = torch.zeros_like(cross_flag, dtype=dtype)
        cross_attn_mask.masked_fill_(cross_flag.logical_not(), float("-inf"))
        masks["cross_attention_mask"] = cross_attn_mask.to(device)
        masks["cross_position_ids"] = torch.cat(
            [noisy_pos, clean_pos], dim=0,
        ).unsqueeze(0)

    return masks


def build_prompt_ids(question: str, tokenizer) -> torch.Tensor:
    """Apply chat template to build prompt input_ids."""
    messages = [{
        "role": "user",
        "content": question + "\nLet's think step by step\n",
    }]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
    else:
        ids = tokenizer(question, return_tensors="pt").input_ids
    return ids


@torch.no_grad()
def t3d_generate(
    model,
    prompt_ids: torch.Tensor,
    masks: Dict[str, torch.Tensor],
    L: int,
    block_size: int,
    gen_length: int,
    max_iter_per_block: int,
    threshold: float,
    mask_token_id: int,
    pad_token_id: int,
    device: str,
) -> torch.Tensor:
    """Block-diffusion generation. Returns response ids (shape [1, gen_length]).

    Per block: iterate up to `max_iter_per_block` times. Each iter does a model.forward
    on the doubled input (clean half = mirror of noisy), takes the noisy-half logits,
    commits masked positions whose softmax peak > threshold. Fallback commit the
    single highest-confidence masked position when nothing passes threshold (this is
    the same guaranteed-progress fallback as our training reveal helper).
    """
    prompt_len = prompt_ids.shape[1]
    full_len = prompt_len + gen_length
    assert full_len <= L, (
        f"prompt + gen_length ({full_len}) > L ({L}); increase --max_seq_len."
    )

    # Initial working sequence: [prompt, MASK*gen_length, pad].
    x = torch.full((1, L), pad_token_id, dtype=torch.long, device=device)
    x[:, :prompt_len] = prompt_ids.to(device)
    x[:, prompt_len:full_len] = mask_token_id

    num_blocks = (gen_length + block_size - 1) // block_size

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_size
        block_end = min(block_start + block_size, full_len)
        block_len = block_end - block_start

        for iter_idx in range(max_iter_per_block):
            # Check if block has any masked positions left.
            current_noisy_block = x[:, block_start:block_end]
            is_masked = (current_noisy_block == mask_token_id)
            if not is_masked.any():
                break

            # Doubled input: clean half = mirror of noisy half.
            doubled_x = torch.cat([x, x], dim=1)

            out = model(
                input_ids=doubled_x,
                use_cache=False,
                output_router_logits=False,
                **masks,
            )
            # logits over the noisy half (first L positions).
            logits_noisy = out.logits[:, :L]

            # Block-only slice.
            block_logits = logits_noisy[:, block_start:block_end].float()
            probs = F.softmax(block_logits, dim=-1)
            max_probs, argmax_ids = probs.max(dim=-1)            # [1, block_len]

            # Threshold reveal -- only commit positions that are still masked.
            confident = (max_probs > threshold) & is_masked

            # Guaranteed progress: if nothing passes, commit the highest-conf masked pos.
            if confident.sum() == 0 and is_masked.any():
                masked_conf = torch.where(
                    is_masked, max_probs, torch.full_like(max_probs, -1.0),
                )
                top_idx = masked_conf.argmax(dim=-1, keepdim=True)     # [1, 1]
                confident = torch.zeros_like(is_masked)
                confident.scatter_(1, top_idx, True)

            new_block = torch.where(confident, argmax_ids, current_noisy_block)
            x[:, block_start:block_end] = new_block

    return x[:, prompt_len:full_len]


def main():
    args = parse_args()

    tok_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    print(f"[gsm8k] tokenizer loaded from {tok_path}")

    model = load_model(args)
    model.eval().to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[gsm8k] model loaded; {n_params/1e9:.2f}B params; on {args.device}")

    # Pre-build masks.
    masks = build_attention_masks(
        args.max_seq_len, args.block_size, args.device, args.model_type,
    )

    # Load dataset.
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Please `pip install datasets` to load GSM8K from HuggingFace.")
    print(f"[gsm8k] loading {args.dataset_name}/{args.split} ...")
    ds = load_dataset(args.dataset_name, "main", split=args.split)
    total = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"[gsm8k] {total} examples to evaluate")

    out_dir = os.path.dirname(args.output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    with open(args.output_path, "w", encoding="utf-8") as fout:
        for idx in range(total):
            example = ds[idx]
            q = example["question"]
            gt = example["answer"]
            prompt_ids = build_prompt_ids(q, tokenizer)

            response_ids = t3d_generate(
                model=model,
                prompt_ids=prompt_ids,
                masks=masks,
                L=args.max_seq_len,
                block_size=args.block_size,
                gen_length=args.gen_length,
                max_iter_per_block=args.max_iter_per_block,
                threshold=args.threshold,
                mask_token_id=args.mask_token_id,
                pad_token_id=args.pad_token_id,
                device=args.device,
            )

            # Detokenize. Strip pad/eos and decode.
            answer_text = tokenizer.decode(
                response_ids[0], skip_special_tokens=True,
            )

            row = {
                "question": q,
                "answer": answer_text,
                "ground_truth": gt,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            elapsed = time.time() - t0
            eta = (total - idx - 1) * elapsed / max(idx + 1, 1)
            print(
                f"[gsm8k {idx+1}/{total}] "
                f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s  "
                f"answer_tail={answer_text[-120:].replace(chr(10), ' ')!r}"
            )

    print(f"[gsm8k] Done. Wrote {total} predictions to {args.output_path}")
    print(f"[gsm8k] Grade with: python tasks/gsm8k_grade.py --pred-path {args.output_path}")


if __name__ == "__main__":
    main()
