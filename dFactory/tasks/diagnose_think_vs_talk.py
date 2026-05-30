"""
Diagnostic v2: think (baseline LLaDA path) vs T3-D talk path, with each path
running under its NATIVE commit rule from iter 1 onward.

Why v2: the original diagnostic fed the same `current_input` to both paths at
each iter (constructed from baseline's prev argmax). That's not what either
model sees at inference -- baseline-style decoding uses LLaDA-2.0-mini's
bundled threshold-OR-topK rule; T3-D will be evaluated under DMax's
`decode_uniform` (left-to-right prefix + top-1 soft embeddings). v2 runs each
path under its native rule so divergence is attributable to talk + delta_head
under realistic decoding, not to an artificial shared trajectory.

Per-iter logic (see token_commit_rule.md for the full mechanism specs):
  Baseline (think -> lm_head, "what an inference run without T3-D would do"):
    - Commit rule = LLaDA-2.0-mini's `model.generate()`:
      threshold-OR-topK with threshold=0.95 (default), num_to_transfer floor.
    - Input at iter k+1: hard tokens; uncommitted positions stay as [MASK].

  T3-D (think -> talk + delta_head -> lm_head, "what eval_t3d_mini.sh runs"):
    - Commit rule = DMax `decode_uniform`:
      left-to-right prefix; cut at first masked position with conf < 0.3.
      Plus the top-1 soft-embedding mix at committed positions, L2-renormalized,
      fed as `inputs_embeds` to talk at the next iter.
    - Both DMax early-stop conditions (all conf >= 0.9, or no change).
    - Anchor cached from iter 0 (think runs ONCE).

End-of-block: convergence summary -- position-level agreement %, KL between
baseline/T3D logits, side-by-side token diff, NFE breakdown.

Healthy signature (a model that passed the leak fix + redesign):
  - T3D argmax changes across iters as soft-embeddings evolve.
  - T3D and baseline disagree at iter 1+ (different commit rules, non-zero
    delta_head).
  - Position-level agreement at end-of-block >= ~85%.
  - T3D's NFE (DMax early-stop) <= baseline's NFE -- ideally a clear win.

Single batch, no kv-cache, no engine optimization. Pure diagnostic.

Usage:
  PYTHONPATH=dFactory:dFactory/VeOmni:$PYTHONPATH \
    python dFactory/tasks/diagnose_think_vs_talk.py \
      --model_path dFactory/outputs/<run>/checkpoints/global_step_<N>/hf_ckpt \
      --tokenizer_path ./LLaDA2.0-mini-moe-merge \
      [--prompt "What is 7 * 8?"] \
      [--gen_length 32] [--n_iters 5]
      [--baseline_threshold 0.95] [--t3d_threshold 0.3]
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "VeOmni")))

from transformers import AutoTokenizer  # noqa: E402

from models.think_talk_llada2.configuration_think_talk_llada2 import (  # noqa: E402
    ThinkTalkLLaDA2Config,
)
from models.think_talk_llada2.modeling_think_talk_llada2 import (  # noqa: E402
    ThinkTalkLLaDA2ForCausalLM,
)

MASK_ID = 156895


# ============================================================================
#                          Attention masks
# ============================================================================

def build_block_causal_mask(L, block_length, dtype, device):
    """4D additive mask [1, 1, L, L]. Position p in block b attends to position
    q in block c iff c <= b. Matches what think saw during T3-D training (the
    noisy-half restriction of the doubled-sequence M_OBC mask) and what DMax's
    `bd_attn_mask` enforces at inference."""
    idx = torch.arange(L, device=device)
    q_block = (idx // block_length).unsqueeze(1)
    kv_block = (idx // block_length).unsqueeze(0)
    allowed = (kv_block <= q_block)
    mask = torch.zeros(1, 1, L, L, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


# ============================================================================
#                          Commit rules
# ============================================================================

def llada2_commit(logits, mask_index, threshold, num_to_transfer):
    """LLaDA-2.0-mini bundled `model.generate()` commit rule.

    Mirrors modeling_llada2_moe.py:1491-1521. Threshold-OR-topK:
      - If >= num_to_transfer positions have confidence > threshold, commit ALL
        of them.
      - Otherwise commit the top-K by confidence (K = num_to_transfer).

    Args:
      logits:        [1, block_length, V]
      mask_index:    [1, block_length] bool, True at currently-MASK positions
      threshold:     float (default 0.95 per LLaDA's bundled config)
      num_to_transfer: int (floor on K; from precomputed schedule)

    Returns:
      x0:            [1, block_length] argmax across vocab
      transfer_index: [1, block_length] bool, True at positions to commit
    """
    x0 = logits.argmax(dim=-1)
    probs = F.softmax(logits.float(), dim=-1)
    max_probs = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    confidence = torch.where(mask_index, max_probs, torch.full_like(max_probs, -float("inf")))

    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
    high_conf_mask = confidence[0] > threshold
    n_high = int(high_conf_mask.sum().item())
    n_active = int(mask_index[0].sum().item())
    if n_high >= num_to_transfer:
        transfer_index[0] = high_conf_mask
    elif n_active > 0:
        k = min(num_to_transfer, n_active)
        _, idx = torch.topk(confidence[0], k=k)
        transfer_index[0, idx] = True
    return x0, transfer_index


def dmax_commit_uniform(logits, mask_index, active_index, threshold):
    """DMax `decode_uniform` commit rule + early-stop detection.

    Mirrors parallel_strategy.py:407-468 (selector) + :564-590 (breakflag).
    Left-to-right prefix cut at first low-confidence masked position; leftmost-
    masked fallback when no candidate qualifies.

    Args:
      logits:       [1, block_length, V]
      mask_index:   [1, block_length] bool, True at currently-MASK positions
      active_index: [1, block_length] bool, True at positions in the decode region
                    (this block's originally-masked positions; stays fixed across iters)
      threshold:    float (DMax inference default 0.3)

    Returns:
      x0:               [1, block_length] argmax
      high_conf_index:  [1, block_length] bool, True at prefix-revealed positions
      max_probs:        [1, block_length] confidence of argmax
      breakflag:        bool, True if early stop fires
      changed_mask:     [1, block_length] bool, True where x0 differs from existing tokens
                        at update positions (used by caller to determine soft-embed refresh)
    """
    x0 = logits.argmax(dim=-1)
    probs = F.softmax(logits.float(), dim=-1)
    max_probs = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)

    confidence = torch.where(mask_index, max_probs, torch.full_like(max_probs, -float("inf")))
    is_low_conf = mask_index & (confidence < threshold)
    has_encountered_failure = torch.cumsum(is_low_conf.long(), dim=1) > 0
    candidates = mask_index & (~has_encountered_failure)

    # Fallback: if no candidate, commit the leftmost masked.
    batch_has_selection = candidates.any(dim=-1, keepdim=True)
    mask_cumsum = torch.cumsum(mask_index.long(), dim=1)
    first_mask_token = (mask_cumsum == 1) & mask_index
    high_conf_index = torch.where(batch_has_selection, candidates, first_mask_token)

    # Early-stop conditions (mirror decode_uniform :579-586):
    # (a) all max_probs in active region >= 0.9, OR (b) no change this step.
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
    """Build inputs_embeds for next iter on T3D path (DMax decode_uniform style).

    Two position classes:
      - Committed (left-to-right prefix above threshold from this or any prior
        iter): SOFT MIX = top1_prob * embed(top1) + (1 - top1_prob) * embed(MASK),
        L2-renormalized. Lets the model revisit committed decisions in later iters.
      - Uncommitted (still masked): hard `embed(MASK)`. The model treats these as
        "still unknown, predict me".

    Stateless: rebuilt from scratch each iter using the current hard token ids
    and the latest logits. Matches parallel_strategy.py:592-668 semantics.

    Args:
      logits:           [1, block_length, V]  -- this iter's talk logits
      max_probs:        [1, block_length]     -- prob of argmax per position
      x0:               [1, block_length]     -- argmax token id per position
      hard_token_ids:   [1, block_length]     -- current hard state (committed positions
                                                 = argmax tokens; uncommitted = mask_id)
      embedding_layer:  nn.Embedding (think's word_embeddings; talk uses same via tied)
      mask_id:          int
      committed_mask:   [1, block_length] bool -- True at committed positions (soft mix)
      uncommitted_mask: [1, block_length] bool -- True at uncommitted positions (hard MASK)
      top_k:            int, default 1 (matches DMax default)

    Returns:
      base_embeds: [1, block_length, D]
    """
    device = logits.device
    dtype = embedding_layer.weight.dtype

    # Start with hard embeds of the current token ids:
    #   - Uncommitted positions hold mask_id, so embedding_layer gives embed(MASK).
    #   - Committed positions hold their argmax (will be overwritten with soft mix).
    base_embeds = embedding_layer(hard_token_ids).clone()

    if not committed_mask.any():
        return base_embeds

    # Soft mix at committed positions (DMax decode_uniform :612-634).
    if top_k == 1:
        topk_probs = max_probs.unsqueeze(-1)       # [B, L, 1]
        topk_indices = x0.unsqueeze(-1)            # [B, L, 1]
    else:
        probs = F.softmax(logits.float(), dim=-1)
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)

    residual_probs = (1.0 - topk_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)   # [B, L, 1]
    topk_embeds = embedding_layer(topk_indices).to(torch.float32)                   # [B, L, K, D]

    mask_embed = embedding_layer(torch.tensor([mask_id], device=device, dtype=torch.long))
    mask_embed = mask_embed.to(torch.float32)                                       # [1, D]
    mask_norm = mask_embed.norm(p=2)                                                # scalar

    topk_weighted = (topk_embeds * topk_probs.unsqueeze(-1)).sum(dim=2)            # [B, L, D]
    mask_weighted = mask_embed.view(1, 1, -1) * residual_probs                     # [B, L, D]
    soft_embeds = topk_weighted + mask_weighted

    # L2 renormalization to the expected mixture norm (DMax :636-656).
    current_norm = soft_embeds.norm(p=2, dim=-1, keepdim=True)
    topk_norms = topk_embeds.norm(p=2, dim=-1)                                     # [B, L, K]
    expected_topk = (topk_norms * topk_probs).sum(dim=-1, keepdim=True)
    expected_mask = mask_norm * residual_probs
    target_norm = expected_topk + expected_mask
    soft_embeds = soft_embeds * (target_norm / (current_norm + 1e-6))
    soft_embeds = soft_embeds.to(dtype)

    base_embeds[committed_mask] = soft_embeds[committed_mask]
    return base_embeds


# ============================================================================
#                          Model loading + forwards
# ============================================================================

def load_model(model_path, device):
    if os.path.isdir(model_path):
        model_path = os.path.abspath(model_path)
    config = ThinkTalkLLaDA2Config.from_pretrained(model_path)
    if not config.model_type.endswith("_veomni"):
        config.model_type = config.model_type + "_veomni"
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


@torch.no_grad()
def baseline_forward(model, input_ids, attn_mask):
    """think(input) -> lm_head. Returns logits [1, L, V]."""
    think_out = model.model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        position_ids=None,
        use_cache=False,
        output_hidden_states=False,
        output_router_logits=False,
        return_dict=True,
    )
    return model.lm_head(think_out.last_hidden_state)


@torch.no_grad()
def t3d_forward_full(model, input_ids, attn_mask, block_start, block_end):
    """Iter-0 T3-D forward: runs think + talk, returns (block_logits, anchor).

    Anchor is cached for subsequent iters' talk-only forwards.
    """
    L = input_ids.shape[1]
    device = input_ids.device

    think_out = model.model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        position_ids=None,
        use_cache=False,
        output_hidden_states=True,
        output_router_logits=False,
        return_dict=True,
    )
    anchor = model.anchor_fuser(think_out.hidden_states)            # [1, L, D]

    block_ids = input_ids[:, block_start:block_end]
    talk_embeds = model.model.word_embeddings(block_ids)
    block_logits = _talk_through_lm_head(
        model, talk_embeds, anchor, block_start, block_end, device,
    )
    return block_logits, anchor


@torch.no_grad()
def t3d_talk_forward_embeds(model, inputs_embeds, anchor_cached, block_start, block_end):
    """Iter-1+ T3-D forward: talk-only with pre-built inputs_embeds + cached anchor.

    inputs_embeds: [1, block_length, D] -- DMax-style soft-embedding mix.
    Returns block logits [1, block_length, V].
    """
    device = inputs_embeds.device
    return _talk_through_lm_head(
        model, inputs_embeds, anchor_cached, block_start, block_end, device,
    )


def _talk_through_lm_head(model, talk_embeds, anchor, block_start, block_end, device):
    """Shared talk forward: takes pre-built embeddings, returns lm_head logits.

    Mirrors the talk + delta_head + lm_head wiring in
    ThinkTalkLLaDA2ForCausalLM.run_talk_block / run_talk (hybrid_xattn mode).
    """
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


# ============================================================================
#                          Reporting
# ============================================================================

def print_compare(label, base_block_ids, t3d_block_ids, decode_region_mask,
                  base_input_block, t3d_input_block, mask_id, tokenizer):
    """Side-by-side per-iter report. Shows each path's input + argmax + count
    of MASK positions remaining."""
    bl = base_block_ids.shape[0]
    n_diff_all = int((base_block_ids != t3d_block_ids).sum().item())
    n_decode = int(decode_region_mask.sum().item())
    n_diff_decode = int(((base_block_ids != t3d_block_ids) & decode_region_mask).sum().item())
    base_mask_remaining = int((base_input_block == mask_id).sum().item())
    t3d_mask_remaining = int((t3d_input_block == mask_id).sum().item())

    print(f"[{label}] decode region: {decode_region_mask.long().tolist()}")
    print(f"[{label}] BASE input:    {base_input_block.tolist()}  (mask count: {base_mask_remaining})")
    print(f"[{label}] T3D  input:    {t3d_input_block.tolist()}  (mask count: {t3d_mask_remaining})")
    print(f"[{label}] BASE argmax:   {base_block_ids.tolist()}")
    print(f"[{label}] T3D  argmax:   {t3d_block_ids.tolist()}")
    print(f"[{label}] divergence: {n_diff_all}/{bl} positions, {n_diff_decode}/{n_decode} at decode-region positions")
    print(f"[{label}] BASE decoded:  {tokenizer.decode(base_block_ids, skip_special_tokens=False)!r}")
    print(f"[{label}] T3D  decoded:  {tokenizer.decode(t3d_block_ids, skip_special_tokens=False)!r}")


def print_convergence_summary(
    base_block_ids_final, t3d_block_ids_final,
    base_logits_final, t3d_logits_final,
    decode_region_mask, base_nfe, t3d_nfe, base_done_at, t3d_done_at, tokenizer,
):
    """End-of-block convergence summary (training_redesign_plan / §7.5)."""
    print("\n" + "=" * 80)
    print("END-OF-BLOCK CONVERGENCE SUMMARY")
    print("=" * 80)

    n_decode = int(decode_region_mask.sum().item())
    if n_decode == 0:
        print("[conv] decode region empty -- nothing to compare")
        return

    agree = (base_block_ids_final == t3d_block_ids_final) & decode_region_mask
    n_agree = int(agree.sum().item())
    agreement = n_agree / n_decode if n_decode else 0.0
    print(f"[conv] position-level agreement (decode region): {n_agree}/{n_decode} = {agreement:.1%}")

    # Per-position KL averaged over decode region.
    # base/t3d_logits_final are [1, block_length, V]; squeeze the batch dim so the
    # [block_length] decode_region_mask can index cleanly.
    base_logp = F.log_softmax(base_logits_final.float(), dim=-1)
    t3d_logp = F.log_softmax(t3d_logits_final.float(), dim=-1)
    kl_per_pos = (base_logp.exp() * (base_logp - t3d_logp)).sum(dim=-1).squeeze(0)  # [L]
    kl_decode = kl_per_pos[decode_region_mask]
    if kl_decode.numel() > 0:
        print(f"[conv] mean KL(BASE || T3D) over decode region: {float(kl_decode.mean().item()):.4f}")
        print(f"[conv] max  KL over decode region:               {float(kl_decode.max().item()):.4f}")

    print(f"[conv] NFE breakdown: BASE talk forwards={base_nfe} (early-stop iter={base_done_at}), "
          f"T3D talk forwards={t3d_nfe} (early-stop iter={t3d_done_at})")
    if t3d_nfe < base_nfe:
        print(f"[conv] T3D NFE win: -{base_nfe - t3d_nfe} forwards (-{(1 - t3d_nfe / max(base_nfe, 1)):.1%})")
    elif t3d_nfe > base_nfe:
        print(f"[conv] T3D NFE regression: +{t3d_nfe - base_nfe} forwards")
    else:
        print(f"[conv] T3D NFE neutral")

    print(f"[conv] BASE final decoded: {tokenizer.decode(base_block_ids_final, skip_special_tokens=False)!r}")
    print(f"[conv] T3D  final decoded: {tokenizer.decode(t3d_block_ids_final, skip_special_tokens=False)!r}")


# ============================================================================
#                          Main
# ============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--prompt", default="What is 7 * 8?")
    p.add_argument("--gen_length", type=int, default=32)
    p.add_argument("--block_length", type=int, default=32,
                   help="MUST match training (v6e: 32).")
    p.add_argument("--n_iters", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--add_chat_template", action="store_true")
    # T3-D v2 ADDED: split commit thresholds.
    p.add_argument("--baseline_threshold", type=float, default=0.95,
                   help="LLaDA-2.0-mini bundled `model.generate()` default = 0.95.")
    p.add_argument("--t3d_threshold", type=float, default=0.3,
                   help="DMax `eval_llada_mini.sh` default = 0.3.")
    p.add_argument("--soft_top_k", type=int, default=1,
                   help="K for DMax soft-embedding mix on T3D path. DMax default = 1.")
    args = p.parse_args()

    tok_path = args.tokenizer_path or args.model_path
    if os.path.isdir(tok_path):
        tok_path = os.path.abspath(tok_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    model = load_model(args.model_path, args.device)

    if args.add_chat_template:
        messages = [{"role": "user", "content": args.prompt + "\nLet's think step by step\n"}]
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt",
        )
    else:
        prompt_ids = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False).input_ids
    prompt_ids = prompt_ids.to(args.device)

    P = prompt_ids.shape[1]
    raw_L = P + args.gen_length
    L = ((raw_L + args.block_length - 1) // args.block_length) * args.block_length
    block_start = (P // args.block_length) * args.block_length
    block_end = block_start + args.block_length

    x = torch.full((1, L), MASK_ID, dtype=torch.long, device=args.device)
    x[:, :P] = prompt_ids

    attn_mask = build_block_causal_mask(L, args.block_length, dtype=torch.bfloat16, device=args.device)

    print(f"[diag v2] mask_id={MASK_ID}  prompt_length={P}  gen_length={args.gen_length}  "
          f"block_length={args.block_length}  L_total={L}")
    print(f"[diag v2] prompt: {tokenizer.decode(prompt_ids[0], skip_special_tokens=False)!r}")
    print(f"[diag v2] first decode block: [{block_start}, {block_end})")
    print(f"[diag v2] thresholds: BASE={args.baseline_threshold}  T3D={args.t3d_threshold}")
    print(f"[diag v2] T3D soft-embed top_k={args.soft_top_k}")
    print(f"[diag v2] delta_head present: {model.delta_head is not None}")

    # Decode region within the block: positions that started as MASK.
    decode_region = (x[0, block_start:block_end] == MASK_ID)

    # LLaDA-2.0 num_to_transfer floor: block_length // n_iters per step.
    num_to_transfer = max(1, args.block_length // max(args.n_iters, 1))
    print(f"[diag v2] BASE num_to_transfer floor (per LLaDA-2.0 schedule): {num_to_transfer}")

    # State per path.
    current_ids_base = x.clone()
    current_ids_t3d = x.clone()
    soft_embeds_t3d = None   # set after iter 0 commit
    anchor_cached = None
    base_done = False
    t3d_done = False
    base_done_at = None
    t3d_done_at = None
    base_nfe = 0
    t3d_nfe = 0

    # Track final logits/ids for convergence summary.
    base_logits_final = None
    t3d_logits_final = None
    base_block_ids_final = None
    t3d_block_ids_final = None

    # ----------------------------------------------------- iter 0
    print("\n" + "=" * 80)
    print("ITER 0  |  both paths see prompt-tail + MASKs  (anchor computed FRESH for T3D)")
    print("=" * 80)

    base_logits = baseline_forward(model, current_ids_base, attn_mask)
    base_block_logits = base_logits[0:1, block_start:block_end]
    base_nfe += 1
    base_block_ids_0 = base_block_logits[0].argmax(dim=-1)

    t3d_block_logits, anchor_cached = t3d_forward_full(
        model, current_ids_t3d, attn_mask, block_start, block_end,
    )
    t3d_nfe += 1
    t3d_block_ids_0 = t3d_block_logits[0].argmax(dim=-1)

    print_compare(
        "iter 0", base_block_ids_0, t3d_block_ids_0, decode_region,
        current_ids_base[0, block_start:block_end],
        current_ids_t3d[0, block_start:block_end],
        MASK_ID, tokenizer,
    )

    base_logits_final = base_block_logits
    t3d_logits_final = t3d_block_logits
    base_block_ids_final = base_block_ids_0
    t3d_block_ids_final = t3d_block_ids_0

    # Apply iter-0 commit on each path.
    # BASE: LLaDA-2.0 commit.
    base_mask_idx = (current_ids_base[0:1, block_start:block_end] == MASK_ID)
    base_x0, base_transfer = llada2_commit(
        base_block_logits, base_mask_idx, args.baseline_threshold, num_to_transfer,
    )
    if base_transfer.any():
        new_block = current_ids_base[0, block_start:block_end].clone()
        new_block[base_transfer[0]] = base_x0[0][base_transfer[0]]
        current_ids_base[0, block_start:block_end] = new_block
    # Block-empty early-stop for BASE.
    if (current_ids_base[0, block_start:block_end] == MASK_ID).sum().item() == 0:
        base_done = True
        base_done_at = 0

    # T3D: DMax commit (left-to-right prefix) + build inputs_embeds for iter 1.
    t3d_mask_idx = (current_ids_t3d[0:1, block_start:block_end] == MASK_ID)
    active_index = decode_region.unsqueeze(0)
    t3d_x0, high_conf_idx, max_probs, breakflag = dmax_commit_uniform(
        t3d_block_logits, t3d_mask_idx, active_index, args.t3d_threshold,
    )
    # Update hard token state: write argmax at committed (high-conf prefix) positions.
    curr_t3d_block = current_ids_t3d[0:1, block_start:block_end]
    update_mask = high_conf_idx | (active_index & ~t3d_mask_idx)
    changed_mask = update_mask & (t3d_x0 != curr_t3d_block)
    if update_mask.any():
        new_block = current_ids_t3d[0, block_start:block_end].clone()
        new_block[update_mask[0]] = t3d_x0[0][update_mask[0]]
        current_ids_t3d[0, block_start:block_end] = new_block
    # Build inputs_embeds for iter 1:
    #   committed positions   -> hard embed of the committed argmax (in current_ids_t3d)
    #   uncommitted positions -> top-1 soft mix + (1 - p) * embed(MASK), L2-renormalized
    new_t3d_mask_idx = (current_ids_t3d[0:1, block_start:block_end] == MASK_ID)
    committed_mask = active_index & (~new_t3d_mask_idx)
    uncommitted_mask = active_index & new_t3d_mask_idx
    soft_embeds_t3d = build_inputs_embeds(
        t3d_block_logits, max_probs, t3d_x0,
        current_ids_t3d[0:1, block_start:block_end],
        model.model.word_embeddings, MASK_ID,
        committed_mask=committed_mask, uncommitted_mask=uncommitted_mask,
        top_k=args.soft_top_k,
    )
    if breakflag:
        t3d_done = True
        t3d_done_at = 0

    # ----------------------------------------------------- iter 1..N
    for k in range(1, args.n_iters + 1):
        print("\n" + "=" * 80)
        print(f"ITER {k}  |  base_done={base_done}  t3d_done={t3d_done}  "
              f"BASE input updated by LLaDA-2.0 commit, T3D by DMax commit + soft embeds")
        print("=" * 80)

        # BASE step.
        if not base_done:
            base_logits = baseline_forward(model, current_ids_base, attn_mask)
            base_block_logits = base_logits[0:1, block_start:block_end]
            base_nfe += 1
            base_block_ids_k = base_block_logits[0].argmax(dim=-1)
            base_logits_final = base_block_logits
            base_block_ids_final = base_block_ids_k
        else:
            base_block_ids_k = base_block_ids_final

        # T3D step (talk-only with soft embeds).
        if not t3d_done:
            t3d_block_logits = t3d_talk_forward_embeds(
                model, soft_embeds_t3d, anchor_cached, block_start, block_end,
            )
            t3d_nfe += 1
            t3d_block_ids_k = t3d_block_logits[0].argmax(dim=-1)
            t3d_logits_final = t3d_block_logits
            t3d_block_ids_final = t3d_block_ids_k
        else:
            t3d_block_ids_k = t3d_block_ids_final

        print_compare(
            f"iter {k}", base_block_ids_k, t3d_block_ids_k, decode_region,
            current_ids_base[0, block_start:block_end],
            current_ids_t3d[0, block_start:block_end],
            MASK_ID, tokenizer,
        )

        # BASE commit + done check.
        if not base_done:
            base_mask_idx = (current_ids_base[0:1, block_start:block_end] == MASK_ID)
            base_x0, base_transfer = llada2_commit(
                base_block_logits, base_mask_idx, args.baseline_threshold, num_to_transfer,
            )
            if base_transfer.any():
                new_block = current_ids_base[0, block_start:block_end].clone()
                new_block[base_transfer[0]] = base_x0[0][base_transfer[0]]
                current_ids_base[0, block_start:block_end] = new_block
            if (current_ids_base[0, block_start:block_end] == MASK_ID).sum().item() == 0:
                base_done = True
                base_done_at = k

        # T3D commit + rebuild inputs_embeds + done check.
        if not t3d_done:
            t3d_mask_idx = (current_ids_t3d[0:1, block_start:block_end] == MASK_ID)
            t3d_x0, high_conf_idx, max_probs, breakflag = dmax_commit_uniform(
                t3d_block_logits, t3d_mask_idx, active_index, args.t3d_threshold,
            )
            curr_t3d_block = current_ids_t3d[0:1, block_start:block_end]
            update_mask = high_conf_idx | (active_index & ~t3d_mask_idx)
            changed_mask = update_mask & (t3d_x0 != curr_t3d_block)
            # Second early-stop: no change this iter.
            if not changed_mask.any():
                breakflag = True
            if update_mask.any():
                new_block = current_ids_t3d[0, block_start:block_end].clone()
                new_block[update_mask[0]] = t3d_x0[0][update_mask[0]]
                current_ids_t3d[0, block_start:block_end] = new_block
            # Rebuild inputs_embeds: committed -> hard embed, uncommitted -> soft mix.
            new_t3d_mask_idx = (current_ids_t3d[0:1, block_start:block_end] == MASK_ID)
            committed_mask = active_index & (~new_t3d_mask_idx)
            uncommitted_mask = active_index & new_t3d_mask_idx
            soft_embeds_t3d = build_inputs_embeds(
                t3d_block_logits, max_probs, t3d_x0,
                current_ids_t3d[0:1, block_start:block_end],
                model.model.word_embeddings, MASK_ID,
                committed_mask=committed_mask, uncommitted_mask=uncommitted_mask,
                top_k=args.soft_top_k,
            )
            if breakflag:
                t3d_done = True
                t3d_done_at = k

        if base_done and t3d_done:
            break

    if base_done_at is None:
        base_done_at = args.n_iters
    if t3d_done_at is None:
        t3d_done_at = args.n_iters

    # End-of-block convergence summary.
    print_convergence_summary(
        base_block_ids_final, t3d_block_ids_final,
        base_logits_final, t3d_logits_final,
        decode_region, base_nfe, t3d_nfe, base_done_at, t3d_done_at, tokenizer,
    )


if __name__ == "__main__":
    main()
