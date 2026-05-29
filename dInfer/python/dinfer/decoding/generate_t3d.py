# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# T3-D block-diffusion decoding. Mirrors DMax's `BlockDiffusionLLM` pattern from
# dInfer/python/dinfer/decoding/generate_uniform.py with the heavy `model(...)`
# forward replaced by `think_once_per_block + talk_N_iters`.
#
# Per-request flow:
#   1. PREFILL: run think on the prompt to build think's KV cache + the prompt's
#      anchor (single forward; chunked if prompt > prefilling_limit).
#   2. For each generation block [block_start, block_end):
#      a. EXTEND-THINK: run think on the (cross-block) segment
#         x[:, block_start-block_length : block_end] (2*block_length tokens)
#         with past_key_values = think's cached KV up to (block_start-block_length).
#         This refreshes the previous block's KV (which was set when the previous
#         block was all-mask; once committed, that KV is stale) and adds KV for the
#         current block. Returns anchor for both blocks; we append to the running
#         anchor tensor.
#         For the first generation block (no previous gen block to refresh), we
#         instead extend only by `block_length` from the cached prompt.
#      b. TALK ITERATIONS: while masks remain in [block_start, block_end), run
#         `forward_talk(x_block, anchor_so_far)` and commit positions via DMax's
#         soft-parallel-decode threshold rule (see `_commit_uniform`).
#   3. Return x[:, prompt_length:].
#
# The reveal rule mirrors `get_transfer_index_uniform` in DMax's
# parallel_strategy.py: find the contiguous high-confidence prefix of masked
# positions (left-to-right), stop at the first sub-threshold position, and
# commit everything in that prefix. Fallback: if no candidate, commit the
# left-most masked position to guarantee progress.

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def build_cross_block_mask(
    past_length: int,
    block_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Block-causal additive mask for the cross-block refresh extend.

    Mirrors DMax's `_get_cross_block_attn_mask` (generate_uniform.py:1276-1289)
    in additive form. The 2-block input is `[prev_block, curr_block]`, length
    `2 * block_length`. The cache covers `[0, past_length)`. The single block
    is: prev-block queries (rows 0..block_length) must NOT see curr-block keys
    (cols past_length+block_length..past_length+2*block_length).

    Returns: [1, 1, 2*block_length, past_length + 2*block_length] additive mask
    (0.0 for allowed, -inf for blocked). The leading two dims broadcast across
    batch and heads.
    """
    q_len = 2 * block_length
    kv_len = past_length + q_len
    mask = torch.zeros(1, 1, q_len, kv_len, dtype=dtype, device=device)
    mask[0, 0, :block_length, past_length + block_length:] = float("-inf")
    return mask


# ============================================================================
#                                 reveal rule
# ============================================================================

def _commit_uniform(
    logits: torch.Tensor,
    block_input: torch.LongTensor,
    mask_id: int,
    threshold: float,
) -> Tuple[torch.LongTensor, torch.BoolTensor]:
    """DMax's soft-parallel-decode rule (parallel_strategy.py:407-468).

    Args:
      logits:      [B, block_length, V] talk's outputs for the current block.
      block_input: [B, block_length] current token state of the block.
      mask_id:     [MASK] token id.
      threshold:   confidence floor for committing a position.

    Returns:
      argmax_ids: [B, block_length] greedy argmax (used to write committed tokens).
      transfer:   [B, block_length] bool; True at positions to be committed this step.
                  The transfer is restricted to a contiguous high-confidence prefix of
                  currently-masked positions. If no position qualifies, the left-most
                  masked position is forced (guaranteed progress).
    """
    probs = F.softmax(logits.float(), dim=-1)
    argmax_ids = probs.argmax(dim=-1)                    # [B, block_length]
    max_probs = probs.gather(-1, argmax_ids.unsqueeze(-1)).squeeze(-1)  # [B, block_length]

    mask_index = (block_input == mask_id)
    confidence = torch.where(mask_index, max_probs, torch.full_like(max_probs, float("-inf")))

    # Mark low-confidence masked positions; cumsum to find first failure cutoff.
    is_low_conf = mask_index & (confidence < threshold)
    has_encountered_failure = torch.cumsum(is_low_conf.long(), dim=1) > 0
    candidates = mask_index & (~has_encountered_failure)

    # Fallback: if a row has no candidate, commit its left-most still-masked position.
    batch_has_selection = candidates.any(dim=-1, keepdim=True)
    mask_cumsum = torch.cumsum(mask_index.long(), dim=1)
    first_masked = (mask_cumsum == 1) & mask_index
    transfer = torch.where(batch_has_selection, candidates, first_masked)

    return argmax_ids, transfer


# ============================================================================
#                                 main loop
# ============================================================================

@dataclass
class T3DGenerateStats:
    """Accounting of model forwards per request. Mirrors DMax's NFE accounting
    but split by which sub-model was invoked, so we can quantify the architecture's
    compute saving."""
    think_forwards: int = 0
    talk_forwards: int = 0


@torch.no_grad()
def generate_t3d(
    inference,                  # ThinkTalkT3DInference
    prompt_ids: torch.LongTensor,   # [B, prompt_length]
    gen_length: int,
    block_length: int,
    threshold: float = 0.9,
    max_iter_per_block: int = 32,
    early_stop: bool = True,
    pad_token_id: Optional[int] = None,
) -> Tuple[torch.LongTensor, T3DGenerateStats]:
    """Block-diffusion decoding for T3-D.

    Args:
      inference:      ThinkTalkT3DInference instance (see modeling_think_talk_t3d.py).
      prompt_ids:     [B, prompt_length] long tensor on inference.device.
      gen_length:     number of response tokens to decode (rounded UP to a multiple
                      of block_length, matching DMax's BlockDiffusionLLM:1308-1310).
      block_length:   per-block decoding window size; matches training's
                      `t3_train_iterations` block split (block_size=32 in v6e).
      threshold:      confidence floor for the soft-parallel-decode reveal.
      max_iter_per_block: hard cap on talk iterations per block (safety net).
      early_stop:     if True, stop generation once an EOS is committed.
      pad_token_id:   filler token for unused tail positions; defaults to model's pad.

    Returns:
      (full_ids, stats). full_ids = [B, prompt_length + gen_length] containing
      prompt + decoded response. stats reports think/talk forwards for the request.
    """
    device = prompt_ids.device
    bsz, prompt_length = prompt_ids.shape[:2]
    pad_id = pad_token_id if pad_token_id is not None else inference.pad_id
    mask_id = inference.mask_id
    eos_id = inference.eos_id

    # Round gen_length up so the response area is a whole number of blocks.
    # Matches DMax's BlockDiffusionLLM:1308 (num_blocks rounds up).
    num_blocks = (gen_length + block_length - 1) // block_length
    actual_gen_length = num_blocks * block_length
    total_length = prompt_length + actual_gen_length

    # Initial working sequence: [prompt, MASK * actual_gen_length].
    x = torch.full((bsz, total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = prompt_ids

    stats = T3DGenerateStats()

    # ------------------------------------------------------------------ prefill
    # Run think on the prompt. Build anchor + KV cache covering [0, prompt_length).
    anchor_so_far, think_kv = inference.prefill_think(x[:, :prompt_length])
    stats.think_forwards += 1
    think_cache_length = prompt_length

    # ------------------------------------------------------------ block decode
    finished = torch.zeros(bsz, dtype=torch.bool, device=device)

    for block_id in range(num_blocks):
        block_start = prompt_length + block_id * block_length
        block_end = block_start + block_length

        # -------- 1. EXTEND THINK + CROSS-BLOCK REFRESH ------------------
        # For block_id == 0, there is no previous gen block to refresh -- we
        # simply extend think by `block_length` tokens (the current block, all
        # mask at this point).
        # For block_id >= 1, the previous gen block's KV was set when that block
        # was all-mask. It is now committed, so its cached K/V is stale. We refresh
        # by re-running think on [prev_block_start, block_end] (length 2 * block_length)
        # with past_kv = think's cache up to prev_block_start. The new anchor for
        # the previous block overwrites our running anchor at those positions.
        if block_id == 0:
            extend_segment = x[:, block_start:block_end]                   # [B, B]
            think_cache_target = block_start
            extend_start_pos = block_start
            anchor_overwrite_start = block_start
            # Single-block extend: all keys are legal targets for the new
            # block's queries (cached prefix + the block itself); SDPA full
            # attention is correct, no explicit mask needed.
            extend_attn_mask = None
        else:
            extend_segment = x[:, block_start - block_length : block_end]  # [B, 2B]
            # Trim cache back to before the stale previous gen block.
            think_kv = inference.trim_cache(think_kv, block_start - block_length)
            think_cache_target = block_start - block_length
            extend_start_pos = block_start - block_length
            anchor_overwrite_start = block_start - block_length
            # Cross-block extend: must block prev-block queries from seeing
            # curr-block keys (which are still all [MASK] at this point).
            # Mirrors DMax's _get_cross_block_attn_mask.
            extend_attn_mask = build_cross_block_mask(
                past_length=block_start - block_length,
                block_length=block_length,
                dtype=anchor_so_far.dtype,
                device=anchor_so_far.device,
            )

        anchor_new, think_kv = inference.extend_think(
            new_segment_ids=extend_segment,
            past_key_values=think_kv,
            start_pos=extend_start_pos,
            attention_mask=extend_attn_mask,
        )
        stats.think_forwards += 1
        think_cache_length = think_cache_target + extend_segment.shape[1]

        # Append/overwrite anchor for the extended range.
        if anchor_overwrite_start == anchor_so_far.shape[1]:
            anchor_so_far = torch.cat([anchor_so_far, anchor_new], dim=1)
        else:
            # block_id >= 1: overwrite the previous block's anchor with the
            # refreshed values, then append the new block's anchor.
            anchor_so_far = torch.cat(
                [anchor_so_far[:, :anchor_overwrite_start, :], anchor_new], dim=1,
            )

        # -------- 2. TALK ITERATIONS --------------------------------------
        for iter_idx in range(max_iter_per_block):
            block_input = x[:, block_start:block_end]
            if not (block_input == mask_id).any():
                break

            logits = inference.forward_talk(
                block_input_ids=block_input,
                anchor_so_far=anchor_so_far,
                block_start=block_start,
                block_end=block_end,
            )
            stats.talk_forwards += 1

            argmax_ids, transfer = _commit_uniform(
                logits=logits,
                block_input=block_input,
                mask_id=mask_id,
                threshold=threshold,
            )
            new_block = torch.where(transfer, argmax_ids, block_input)

            # Avoid committing the mask token id itself.
            new_block = torch.where(
                transfer & (argmax_ids == mask_id), block_input, new_block,
            )
            x[:, block_start:block_end] = new_block

        # -------- 3. EARLY STOP -------------------------------------------
        if early_stop:
            # A row "finishes" once it produces an EOS anywhere in (prompt..now).
            response_so_far = x[:, prompt_length:block_end]
            has_eos = (response_so_far == eos_id).any(dim=1)
            finished = finished | has_eos
            if finished.all():
                # Fill remaining positions in the unfinished rows' response with pad
                # so the decoder doesn't see leftover MASKs.
                x[:, block_end:] = pad_id
                break

    return x, stats


# ============================================================================
#                                  __main__
# ============================================================================

def _smoke_test():
    """Tiny end-to-end smoke test invoked via:
        python -m dinfer.decoding.generate_t3d \
            --model_path /path/to/t3d/checkpoint \
            --tokenizer_path /path/to/tokenizer
    Useful to confirm the loop runs without exceptions on a real checkpoint
    before plugging into lm-eval-harness."""
    import argparse
    from dinfer.model.modeling_think_talk_t3d import ThinkTalkT3DInference

    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--prompt", default="What is 7 * 8?\nLet's think step by step\n")
    p.add_argument("--gen_length", type=int, default=128)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.9)
    args = p.parse_args()

    inf = ThinkTalkT3DInference(args.model_path, args.tokenizer_path)
    messages = [{"role": "user", "content": args.prompt}]
    prompt_ids = inf.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt",
    ).to(inf.device)
    out, stats = generate_t3d(
        inf, prompt_ids,
        gen_length=args.gen_length,
        block_length=args.block_length,
        threshold=args.threshold,
    )
    text = inf.tokenizer.decode(
        out[0, prompt_ids.shape[1]:], skip_special_tokens=True,
    )
    print(f"[T3-D] think={stats.think_forwards} talk={stats.talk_forwards}")
    print(f"[T3-D] answer: {text}")


if __name__ == "__main__":
    _smoke_test()
