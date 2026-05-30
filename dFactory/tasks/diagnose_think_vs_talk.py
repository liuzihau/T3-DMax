"""
Diagnostic: think (baseline LLaDA path) vs T3-D talk path, iter-by-iter.

For each iteration k in [0, N]:
  - input_k = prompt + (baseline_iter_{k-1} argmax, or MASK*gen_length at k=0)
  - baseline_k = argmax( lm_head( think(input_k).last_hidden ) ) [over response area]
  - t3d_k      = argmax( lm_head( anchor_0 + delta_head( talk(input_k, anchor_0) ) ) )

Note that anchor is COMPUTED FRESH AT iter 0 ONLY and CACHED — every subsequent
iter reuses iter_0's anchor (which matches training's design: think once per block).

The input at each iter is taken from the BASELINE's previous output (not T3D's),
so both paths see identical inputs at iter k>0 and any divergence is attributable
to talk + delta_head, not to differing trajectories.

What to look for:
  - If baseline_0 produces sensible tokens but t3d_0 is degenerate -> talk model
    is broken / delta_head is broken / anchor wiring is broken.
  - If baseline_0 is also garbage -> the think backbone itself is broken (e.g.,
    lm_head got drift, embedding got drift, or model weights are corrupted).
  - If baseline_0 sensible AND t3d_0 sensible but t3d_k diverges at higher k ->
    talk has not learned to handle progressively-revealed inputs (the multi-iter
    A4 training is failing to give it that capability).

Single batch, no optimization, no kv-cache. Pure diagnostic.

Usage:
  PYTHONPATH=dFactory:dFactory/VeOmni:$PYTHONPATH \
    python dFactory/tasks/diagnose_think_vs_talk.py \
      --model_path dFactory/outputs/<run>/checkpoints/global_step_<N>/hf_ckpt \
      --tokenizer_path ./LLaDA2.0-mini-moe-merge \
      [--prompt "What is 7 * 8?"] \
      [--gen_length 32] [--n_iters 5]
"""

import argparse
import os
import sys

import torch

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


def build_block_causal_mask(L, block_length, dtype, device):
    """4D additive mask [1, 1, L, L]. Position p in block b attends to position
    q in block c iff c <= b. Matches what think saw during T3-D training (the
    noisy-half restriction of the doubled-sequence M_OBC mask) and what DMax's
    `bd_attn_mask` enforces at inference (generate_uniform.py:1254-1265)."""
    idx = torch.arange(L, device=device)
    q_block = (idx // block_length).unsqueeze(1)   # [L, 1]
    kv_block = (idx // block_length).unsqueeze(0)  # [1, L]
    allowed = (kv_block <= q_block)                # [L, L] bool
    mask = torch.zeros(1, 1, L, L, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


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
    """think(input) -> lm_head(last_hidden). Returns logits [1, L, V].

    `attn_mask` must be the block-causal 4D additive mask used at training.
    Passing None gives full bidirectional attention, which is out-of-
    distribution for LLaDA-2.0-mini frozen think weights."""
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
def t3d_forward(model, input_ids, attn_mask, anchor_cached=None):
    """T3-D path on single-L input. Returns (logits, anchor).

    If anchor_cached is None: compute fresh anchor from think on this input
    (with the block-causal `attn_mask`).
    Otherwise: skip think; use the provided cached anchor and only run talk.

    Talk's self-attn at inference is block-diagonal (within-block bidirectional)
    -- matches the L self-attn mask used at training. We rebuild it here.
    Cross-attn from talk noisy-q to the (single-L) anchor uses block-causal
    too -- at inference there's no clean half, so the L-by-2L training mask
    collapses to the same block-causal L-by-L pattern.
    """
    L = input_ids.shape[1]
    device = input_ids.device

    if anchor_cached is None:
        think_out = model.model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=None,
            use_cache=False,
            output_hidden_states=True,
            output_router_logits=False,
            return_dict=True,
        )
        anchor = model.anchor_fuser(think_out.hidden_states)
    else:
        anchor = anchor_cached

    pos = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0)
    talk_embeds = model.model.word_embeddings(input_ids)

    # Talk self-attn mask: block-diagonal (each block sees only itself,
    # bidirectional within). Cross-attn mask: block-causal (block b sees
    # anchor positions at blocks <= b). We derive both from `attn_mask`
    # (which is the block-causal mask the caller built for think).
    if attn_mask is not None:
        allowed_bc = attn_mask[0, 0].eq(0)                  # [L, L] bool, True=allowed
        block_diag = allowed_bc & allowed_bc.T              # symmetric closure = block-diag
        self_attn_mask = torch.zeros(1, 1, L, L, dtype=attn_mask.dtype, device=device)
        self_attn_mask.masked_fill_(~block_diag, float("-inf"))
    else:
        self_attn_mask = None

    talk_hidden = model.talk_model(
        inputs_embeds=talk_embeds,
        anchor=anchor,
        attention_mask=self_attn_mask,
        position_ids=pos,
        anchor_kv=anchor,
        cross_attention_mask=attn_mask,    # block-causal L-by-L
        cross_position_ids=pos,
    )

    if model.delta_head is not None:
        talk_hidden = anchor + model.delta_head(talk_hidden)
    elif getattr(model.config, "add_anchor_skip_residual", False):
        talk_hidden = talk_hidden + anchor

    return model.lm_head(talk_hidden), anchor


def print_compare(label, base_ids, t3d_ids, tokenizer, n_show=32):
    n_diff = int((base_ids != t3d_ids).sum().item())
    n_total = base_ids.shape[0]
    print(f"[{label}] divergence: {n_diff}/{n_total} positions differ between baseline and t3d")
    print(f"[{label}] baseline IDs[:{n_show}]: {base_ids[:n_show].tolist()}")
    print(f"[{label}]      t3d IDs[:{n_show}]: {t3d_ids[:n_show].tolist()}")
    base_text = tokenizer.decode(base_ids, skip_special_tokens=False)
    t3d_text = tokenizer.decode(t3d_ids, skip_special_tokens=False)
    print(f"[{label}] baseline decoded: {base_text!r}")
    print(f"[{label}]      t3d decoded: {t3d_text!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--prompt", default="What is 7 * 8?")
    p.add_argument("--gen_length", type=int, default=32)
    p.add_argument("--block_length", type=int, default=32,
                   help="MUST match training (v6e: 32). Used to build the block-causal mask.")
    p.add_argument("--n_iters", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--add_chat_template", action="store_true",
                   help="Wrap prompt in tokenizer.apply_chat_template (matches training data format)")
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
    # Round L up to a multiple of block_length so the block-causal mask is well-defined.
    raw_L = P + args.gen_length
    L = ((raw_L + args.block_length - 1) // args.block_length) * args.block_length

    # Initial sequence: prompt + MASK * (L - prompt_length).
    x = torch.full((1, L), MASK_ID, dtype=torch.long, device=args.device)
    x[:, :P] = prompt_ids

    attn_mask = build_block_causal_mask(L, args.block_length, dtype=torch.bfloat16, device=args.device)

    print(f"[diag] mask_id={MASK_ID}  prompt_length={P}  gen_length={args.gen_length}  block_length={args.block_length}  L_total={L}")
    print(f"[diag] prompt: {tokenizer.decode(prompt_ids[0], skip_special_tokens=False)!r}")
    print(f"[diag] delta_head present: {model.delta_head is not None}")
    print(f"[diag] add_anchor_skip_residual: {getattr(model.config, 'add_anchor_skip_residual', False)}")
    print(f"[diag] attn_mask: block-causal at block_length={args.block_length}  (matches training)")

    # ------------------------------------------------------------------ iter 0
    print("\n" + "=" * 80)
    print("ITER 0  |  input = prompt + MASK*gen_length  (anchor is computed FRESH here)")
    print("=" * 80)

    base_logits = baseline_forward(model, x, attn_mask)
    base_ids_iter0 = base_logits[0, P:].argmax(dim=-1)

    t3d_logits, anchor_cached = t3d_forward(model, x, attn_mask, anchor_cached=None)
    t3d_ids_iter0 = t3d_logits[0, P:].argmax(dim=-1)

    print_compare("iter 0", base_ids_iter0, t3d_ids_iter0, tokenizer)

    # ------------------------------------------------------------------ iters 1..N
    # The next iter's input is the BASELINE's previous argmax — same input for
    # both paths so divergence at iter k > 0 is attributable to talk/delta_head.
    current_input = x.clone()
    current_input[0, P:] = base_ids_iter0

    for k in range(1, args.n_iters + 1):
        print("\n" + "=" * 80)
        print(f"ITER {k}  |  input = prompt + baseline_iter{k-1}_argmax  (anchor REUSED from iter 0)")
        print("=" * 80)

        base_logits = baseline_forward(model, current_input, attn_mask)
        base_ids = base_logits[0, P:].argmax(dim=-1)

        t3d_logits, _ = t3d_forward(model, current_input, attn_mask, anchor_cached=anchor_cached)
        t3d_ids = t3d_logits[0, P:].argmax(dim=-1)

        print_compare(f"iter {k}", base_ids, t3d_ids, tokenizer)

        # Advance: use this iter's baseline argmax for the next iter
        current_input[0, P:] = base_ids


if __name__ == "__main__":
    main()
