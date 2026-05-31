# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# T3-D block-diffusion decoding — CANONICAL paradigm (the DMax analogue of
# generate_uniform.py's BlockDiffusionLLM, with the heavy per-iter forward
# replaced by think-once-per-block + lightweight talk-N-iters).
#
# This module is the single source of truth for how T3-D decodes. The decode
# math is lifted FAITHFULLY from `dFactory/tasks/diagnose_think_vs_talk.py`
# (the v2 diagnostic the user designated canonical) — same primitives, so the
# accuracy path and the diagnostic cannot drift apart. Keep the two in sync; if
# the decode rule changes, change it here and mirror into the diagnostic.
#
# It SUPERSEDES the previous generate_t3d.py, which diverged from DMax in two
# ways: prompt-relative blocks (vs grid-aligned) and hard-token argmax reveal
# (vs DMax's soft-embedding decode_uniform). Both are fixed here.
#
# Per request (grid-aligned, no KV cache — plain-PyTorch V1 correctness path;
# the vllm port is the eventual throughput path):
#   first_block_start = (P // block) * block      # the first decode block overlaps the prompt tail
#   for each block [bs, be):
#     think ONCE on x[:, :be] (block-causal) -> anchor over [0, be)     [1 think forward]
#     talk iterates with DMax decode_uniform soft-embedding reveal      [N talk forwards]
#       iter 0  : talk on the all-mask block
#       iter k>0: talk on inputs_embeds = top1_prob*embed(top1) + (1-p)*embed(MASK), L2-renorm
#       commit  : left-to-right high-confidence prefix (dmax_commit_uniform)
#       stop    : all active conf >= 0.9, or no change, or block fully committed

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Make the dFactory ThinkTalk model importable for load_t3d_model (lazy import
# below). The decode primitives themselves need only torch, so importing this
# module for them never requires the model package.
_HERE = os.path.dirname(os.path.abspath(__file__))                      # .../dInfer/python/dinfer/decoding
_T3DMAX_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))  # .../T3-DMax
_DFACTORY = os.path.join(_T3DMAX_ROOT, "dFactory")
for _p in (_DFACTORY, os.path.join(_DFACTORY, "VeOmni")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

MASK_ID = 156895
EOS_ID = 156892
PAD_ID = 156892


# ============================================================================
#                          attention masks
# ============================================================================

def build_block_causal_mask(L, block_length, dtype, device):
    """4D additive mask [1, 1, L, L]. Position p in block b attends to position q
    in block c iff c <= b. Matches T3-D training's M_OBC (noisy-half restriction)
    and DMax's `bd_attn_mask` at inference."""
    idx = torch.arange(L, device=device)
    q_block = (idx // block_length).unsqueeze(1)
    kv_block = (idx // block_length).unsqueeze(0)
    allowed = (kv_block <= q_block)
    mask = torch.zeros(1, 1, L, L, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


# ============================================================================
#                          commit rule (DMax decode_uniform)
# ============================================================================

def dmax_commit_uniform(logits, mask_index, active_index, threshold):
    """DMax `decode_uniform` commit rule + early-stop detection.

    Mirrors parallel_strategy.py:407-468 (selector) + :564-590 (breakflag).
    Left-to-right prefix cut at the first low-confidence masked position;
    leftmost-masked fallback when no candidate qualifies.

    Returns: (x0, high_conf_index, max_probs, breakflag).
    """
    x0 = logits.argmax(dim=-1)
    probs = F.softmax(logits.float(), dim=-1)
    max_probs = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

    confidence = torch.where(mask_index, max_probs, torch.full_like(max_probs, -float("inf")))
    is_low_conf = mask_index & (confidence < threshold)
    has_encountered_failure = torch.cumsum(is_low_conf.long(), dim=1) > 0
    candidates = mask_index & (~has_encountered_failure)

    batch_has_selection = candidates.any(dim=-1, keepdim=True)
    mask_cumsum = torch.cumsum(mask_index.long(), dim=1)
    first_mask_token = (mask_cumsum == 1) & mask_index
    high_conf_index = torch.where(batch_has_selection, candidates, first_mask_token)

    breakflag = False
    if active_index.any():
        if bool((max_probs[active_index] >= 0.9).all().item()):
            breakflag = True

    return x0, high_conf_index, max_probs, breakflag


def build_inputs_embeds(
    logits, max_probs, x0, hard_token_ids,
    embedding_layer, mask_id,
    committed_mask, uncommitted_mask, top_k=1,
):
    """DMax decode_uniform soft-embedding mix for the next iter's talk input.

    Committed positions: top1_prob*embed(top1) + (1-top1_prob)*embed(MASK),
    L2-renormalized. Uncommitted positions: hard embed(MASK). Stateless; rebuilt
    each iter. Mirrors parallel_strategy.py:592-668.
    """
    device = logits.device
    dtype = embedding_layer.weight.dtype

    base_embeds = embedding_layer(hard_token_ids).clone()
    if not committed_mask.any():
        return base_embeds

    if top_k == 1:
        topk_probs = max_probs.unsqueeze(-1)
        topk_indices = x0.unsqueeze(-1)
    else:
        probs = F.softmax(logits.float(), dim=-1)
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)

    residual_probs = (1.0 - topk_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    topk_embeds = embedding_layer(topk_indices).to(torch.float32)

    mask_embed = embedding_layer(torch.tensor([mask_id], device=device, dtype=torch.long)).to(torch.float32)
    mask_norm = mask_embed.norm(p=2)

    topk_weighted = (topk_embeds * topk_probs.unsqueeze(-1)).sum(dim=2)
    mask_weighted = mask_embed.view(1, 1, -1) * residual_probs
    soft_embeds = topk_weighted + mask_weighted

    current_norm = soft_embeds.norm(p=2, dim=-1, keepdim=True)
    topk_norms = topk_embeds.norm(p=2, dim=-1)
    expected_topk = (topk_norms * topk_probs).sum(dim=-1, keepdim=True)
    expected_mask = mask_norm * residual_probs
    target_norm = expected_topk + expected_mask
    soft_embeds = soft_embeds * (target_norm / (current_norm + 1e-6))
    soft_embeds = soft_embeds.to(dtype)

    base_embeds[committed_mask] = soft_embeds[committed_mask]
    return base_embeds


# ============================================================================
#                          model loading + forwards
# ============================================================================

def load_t3d_model(model_path, device="cuda"):
    """Load ThinkTalkLLaDA2ForCausalLM (raw model; the canonical decode operates
    on it directly — no inference shim). Mirrors diagnose_think_vs_talk.load_model."""
    from models.think_talk_llada2.configuration_think_talk_llada2 import ThinkTalkLLaDA2Config
    from models.think_talk_llada2.modeling_think_talk_llada2 import ThinkTalkLLaDA2ForCausalLM

    # Always normalize: HF rejects any path containing '..' (it treats it as a
    # repo id -> HFValidationError). abspath collapses '..' lexically.
    model_path = os.path.abspath(model_path)
    config = ThinkTalkLLaDA2Config.from_pretrained(model_path)
    if not str(config.model_type).endswith("_veomni"):
        config.model_type = str(config.model_type) + "_veomni"
    if getattr(config, "moe_implementation", None) != "fused":
        config.moe_implementation = "fused"
    model = ThinkTalkLLaDA2ForCausalLM.from_pretrained(
        model_path, config=config,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    if hasattr(model.model, "gradient_checkpointing"):
        model.model.gradient_checkpointing = False
    model.eval().to(device)
    return model


def _talk_through_lm_head(model, talk_embeds, anchor, block_start, block_end, device):
    """Shared talk forward: pre-built embeddings -> lm_head logits. Mirrors the
    talk + delta_head + lm_head wiring in run_talk_block (hybrid_xattn mode)."""
    pos_self = torch.arange(block_start, block_end, device=device, dtype=torch.long).unsqueeze(0)
    anchor_block = anchor[:, block_start:block_end, :].contiguous()
    anchor_kv = anchor[:, :block_end, :].contiguous()
    pos_cross_kv = torch.arange(0, block_end, device=device, dtype=torch.long).unsqueeze(0)

    talk_hidden = model.talk_model(
        inputs_embeds=talk_embeds,
        anchor=anchor_block,
        attention_mask=None,
        position_ids=pos_self,
        anchor_kv=anchor_kv,
        cross_attention_mask=None,
        cross_position_ids=pos_cross_kv,
    )
    if model.delta_head is not None:
        talk_hidden = anchor_block + model.delta_head(talk_hidden)
    elif getattr(model.config, "add_anchor_skip_residual", False):
        talk_hidden = talk_hidden + anchor_block

    return model.lm_head(talk_hidden)


@torch.no_grad()
def t3d_forward_full(model, input_ids, attn_mask, block_start, block_end):
    """Iter-0 forward: think (-> anchor over [0, L)) + talk on the block.
    Returns (block_logits, anchor). Anchor is cached for the block's later iters."""
    device = input_ids.device
    think_out = model.model(
        input_ids=input_ids, attention_mask=attn_mask, position_ids=None,
        use_cache=False, output_hidden_states=True, output_router_logits=False,
        return_dict=True,
    )
    anchor = model.anchor_fuser(think_out.hidden_states)            # [1, L, D]
    block_ids = input_ids[:, block_start:block_end]
    talk_embeds = model.model.word_embeddings(block_ids)
    block_logits = _talk_through_lm_head(model, talk_embeds, anchor, block_start, block_end, device)
    return block_logits, anchor


@torch.no_grad()
def t3d_talk_forward_embeds(model, inputs_embeds, anchor_cached, block_start, block_end):
    """Iter-1+ forward: talk-only with pre-built soft inputs_embeds + cached anchor."""
    return _talk_through_lm_head(model, inputs_embeds, anchor_cached, block_start, block_end,
                                 inputs_embeds.device)


# ============================================================================
#                          decode
# ============================================================================

@dataclass
class T3DGenerateStats:
    """Forward accounting split by sub-model, to quantify the compute saving."""
    think_forwards: int = 0
    talk_forwards: int = 0


@torch.no_grad()
def decode_block_t3d(model, x, block_start, block_end, attn_mask,
                     threshold, max_iters, soft_top_k):
    """Decode one block: think once (anchor) + talk iters with DMax decode_uniform
    soft-embedding reveal. Mutates x[block] in place. Returns talk_forwards."""
    active_index = (x[0:1, block_start:block_end] == MASK_ID)       # decode region

    block_logits, anchor = t3d_forward_full(model, x[:, :block_end], attn_mask, block_start, block_end)
    talk_forwards = 1

    def _commit_and_build_embeds(logits):
        mask_idx = (x[0:1, block_start:block_end] == MASK_ID)
        x0, high_conf_idx, max_probs, breakflag = dmax_commit_uniform(
            logits, mask_idx, active_index, threshold)
        curr_block = x[0:1, block_start:block_end]
        update_mask = high_conf_idx | (active_index & ~mask_idx)
        changed = update_mask & (x0 != curr_block)
        if update_mask.any():
            nb = x[0, block_start:block_end].clone()
            nb[update_mask[0]] = x0[0][update_mask[0]]
            x[0, block_start:block_end] = nb
        new_mask_idx = (x[0:1, block_start:block_end] == MASK_ID)
        committed_mask = active_index & (~new_mask_idx)
        uncommitted_mask = active_index & new_mask_idx
        soft_embeds = build_inputs_embeds(
            logits, max_probs, x0, x[0:1, block_start:block_end],
            model.model.word_embeddings, MASK_ID,
            committed_mask=committed_mask, uncommitted_mask=uncommitted_mask, top_k=soft_top_k)
        done = bool(breakflag) or (not changed.any()) or (not new_mask_idx.any())
        return done, soft_embeds

    done, soft_embeds = _commit_and_build_embeds(block_logits)
    it = 1
    while not done and it <= max_iters:
        block_logits = t3d_talk_forward_embeds(model, soft_embeds, anchor, block_start, block_end)
        talk_forwards += 1
        done, soft_embeds = _commit_and_build_embeds(block_logits)
        it += 1

    # Safety: never leak a [MASK] into the output.
    still_masked = (x[0:1, block_start:block_end] == MASK_ID)
    if still_masked.any():
        fill = block_logits[0].argmax(dim=-1)
        nb = x[0, block_start:block_end].clone()
        nb[still_masked[0]] = fill[still_masked[0]]
        x[0, block_start:block_end] = nb

    return talk_forwards


@torch.no_grad()
def generate_t3d(model, prompt_ids, gen_length, block_length,
                 threshold=0.3, max_iter_per_block=32, soft_top_k=1,
                 early_stop=True):
    """Grid-aligned multi-block T3-D generation on the raw ThinkTalk model.

    Returns (response_ids [n], T3DGenerateStats). response_ids excludes the
    prompt and is cut at the first EOS.
    """
    device = prompt_ids.device
    P = prompt_ids.shape[1]

    first_block_start = (P // block_length) * block_length
    end_target = P + gen_length
    num_blocks = (end_target - first_block_start + block_length - 1) // block_length
    L = first_block_start + num_blocks * block_length

    x = torch.full((1, L), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids

    stats = T3DGenerateStats()
    eos_cut = L

    for b in range(num_blocks):
        bs = first_block_start + b * block_length
        be = bs + block_length
        attn = build_block_causal_mask(be, block_length, dtype=torch.bfloat16, device=device)
        t_fwd = decode_block_t3d(model, x, bs, be, attn,
                                 threshold=threshold, max_iters=max_iter_per_block,
                                 soft_top_k=soft_top_k)
        stats.think_forwards += 1
        stats.talk_forwards += t_fwd

        if early_stop:
            resp_lo = max(P, bs)
            seg = x[0, resp_lo:be]
            eos_pos = (seg == EOS_ID).nonzero(as_tuple=True)[0]
            if eos_pos.numel() > 0:
                eos_cut = resp_lo + int(eos_pos[0].item())
                if be < L:
                    x[0, be:] = PAD_ID
                break

    return x[0, P:eos_cut].clone(), stats


# ============================================================================
#                          __main__ smoke
# ============================================================================

def _smoke_test():
    """python -m dinfer.decoding.generate_t3d --model_path <ckpt> [--tokenizer_path <tok>]"""
    import argparse
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--prompt", default="What is 7 * 8?")
    p.add_argument("--gen_length", type=int, default=128)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    tok_path = args.tokenizer_path or args.model_path
    if os.path.isdir(tok_path):
        tok_path = os.path.abspath(tok_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    model = load_t3d_model(args.model_path, args.device)
    print(f"[T3-D] delta_head present: {model.delta_head is not None}")

    messages = [{"role": "user", "content": args.prompt + "\nLet's think step by step\n"}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt",
    ).to(args.device)

    response_ids, stats = generate_t3d(
        model, prompt_ids, gen_length=args.gen_length, block_length=args.block_length,
        threshold=args.threshold,
    )
    text = tokenizer.decode(response_ids, skip_special_tokens=True)
    print(f"[T3-D] think_forwards={stats.think_forwards} talk_forwards={stats.talk_forwards}")
    print(f"[T3-D] answer: {text!r}")


if __name__ == "__main__":
    _smoke_test()
