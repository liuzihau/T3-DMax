# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# PREMISE PROBE — stage 1: collection.
#
# Tests the T3-D hypothesis: does a heavy dLLM's FIRST forward already encode the
# answer its iterative decode converges to? Runs a FROZEN full model (point
# --model_path at the DMax-finetuned checkpoint = the proposed T3-D backbone)
# through the canonical block-diffusion decode on GSM8K prompts, caching per
# decode position:
#   h0           : iter-0 last hidden state (what T3-D caches as the anchor)
#   iter0_argmax : lm_head(h0).argmax  -- the one-shot guess
#   y_converged  : the token the full iterative decode commits
#   group        : prompt index (prompt-disjoint split in stage 2)
#
# GSM8K prompts (not the saturated SFT data) give the metric real headroom -- the
# "merge issue C" point. Standard freeze-after-commit decode (re-embed committed
# ids each iter) -> depends only on a plain `.model` + `.lm_head` forward.
#
# Grade/analyze with probe_fit.py.

import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DINFER_PYTHON = os.path.abspath(os.path.join(_HERE, "..", "python"))   # .../dInfer/python
if _DINFER_PYTHON not in sys.path:
    sys.path.insert(0, _DINFER_PYTHON)

from transformers import AutoTokenizer  # noqa: E402

# Canonical mask + commit rule. Importing generate_t3d also puts dFactory on the
# path (its module-level setup), which the fallback loader below relies on.
from dinfer.decoding.generate_t3d import (  # noqa: E402
    EOS_ID,
    MASK_ID,
    build_block_causal_mask,
    dmax_commit_uniform,
)

GSM8K_USER_TEMPLATE = "Question: {question}\nLet's think step by step\nAnswer:"


def load_full_model(path, device):
    """Plain LLaDA2/DMax CausalLM exposing `.model` (-> last_hidden_state) and
    `.lm_head`. AutoModelForCausalLM first; falls back to dFactory LLaDA2MoeModelLM."""
    if os.path.isdir(path):
        path = os.path.abspath(path)
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        )
    except Exception as exc:  # pragma: no cover - depends on checkpoint format
        print(f"[probe] AutoModelForCausalLM failed ({type(exc).__name__}: {exc}); "
              f"falling back to dFactory LLaDA2MoeModelLM")
        from transformers import AutoConfig
        from models.llada2_moe.modeling_llada2_moe import LLaDA2MoeModelLM
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        if not str(getattr(config, "model_type", "")).endswith("_veomni"):
            config.model_type = str(config.model_type) + "_veomni"
        if getattr(config, "moe_implementation", None) != "fused":
            config.moe_implementation = "fused"
        model = LLaDA2MoeModelLM.from_pretrained(
            path, config=config, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if hasattr(getattr(model, "model", None), "gradient_checkpointing"):
        model.model.gradient_checkpointing = False
    model.eval().to(device)
    assert hasattr(model, "model") and hasattr(model, "lm_head"), \
        "loaded model must expose .model and .lm_head"
    return model


@torch.no_grad()
def _think(model, ids, attn_mask):
    out = model.model(
        input_ids=ids, attention_mask=attn_mask, position_ids=None,
        use_cache=False, output_hidden_states=False, return_dict=True,
    )
    return out.last_hidden_state


@torch.no_grad()
def collect_for_prompt(model, prompt_ids, gen_length, block_length, threshold,
                       max_iters_per_block, max_blocks):
    device = prompt_ids.device
    P = prompt_ids.shape[1]
    first_block_start = (P // block_length) * block_length
    end_target = P + gen_length
    num_blocks = (end_target - first_block_start + block_length - 1) // block_length
    num_blocks = min(num_blocks, max_blocks)
    L = first_block_start + num_blocks * block_length

    x = torch.full((1, L), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids

    h0_list, arg0_list, y_list = [], [], []
    for b in range(num_blocks):
        bs = first_block_start + b * block_length
        be = bs + block_length
        attn = build_block_causal_mask(be, block_length, dtype=torch.bfloat16, device=device)
        active_index = (x[0:1, bs:be] == MASK_ID)
        if not active_index.any():
            continue

        h0_block = arg0_block = last_logits = None
        for it in range(max_iters_per_block + 1):
            hidden = _think(model, x[:, :be], attn)
            block_hidden = hidden[:, bs:be]
            logits_block = model.lm_head(block_hidden)
            last_logits = logits_block
            if it == 0:
                h0_block = block_hidden[0].float().cpu()
                arg0_block = logits_block[0].argmax(dim=-1).cpu()

            mask_idx = (x[0:1, bs:be] == MASK_ID)
            x0, high_conf_idx, max_probs, breakflag = dmax_commit_uniform(
                logits_block, mask_idx, active_index, threshold)
            # Standard masked-diffusion: commit newly-revealed prefix and FREEZE.
            update_mask = high_conf_idx
            changed = update_mask & (x0 != x[0:1, bs:be])
            if update_mask.any():
                nb = x[0, bs:be].clone()
                nb[update_mask[0]] = x0[0][update_mask[0]]
                x[0, bs:be] = nb
            if bool(breakflag) or (not changed.any()) or (not (x[0:1, bs:be] == MASK_ID).any()):
                break

        leftover = (x[0:1, bs:be] == MASK_ID)
        if leftover.any() and last_logits is not None:
            fill = last_logits[0].argmax(dim=-1)
            nb = x[0, bs:be].clone()
            nb[leftover[0]] = fill[leftover[0]]
            x[0, bs:be] = nb

        sel = active_index[0].cpu()
        h0_list.append(h0_block[sel])
        arg0_list.append(arg0_block[sel])
        y_list.append(x[0, bs:be].cpu()[sel])

        resp_lo = max(P, bs)
        if (x[0, resp_lo:be] == EOS_ID).any():
            break

    if not h0_list:
        return None
    return (torch.cat(h0_list, 0), torch.cat(arg0_list, 0), torch.cat(y_list, 0))


def main():
    p = argparse.ArgumentParser(description="Premise probe — collect iter-0 hidden + converged tokens")
    p.add_argument("--model_path", required=True,
                   help="FROZEN full model (DMax-finetuned checkpoint = proposed backbone).")
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--out_path", required=True)
    p.add_argument("--num_prompts", type=int, default=200)
    p.add_argument("--gen_length", type=int, default=256)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--max_iters_per_block", type=int, default=32)
    p.add_argument("--max_blocks", type=int, default=8)
    p.add_argument("--gt_jsonl_path", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    tok_path = args.tokenizer_path or args.model_path
    if os.path.isdir(tok_path):
        tok_path = os.path.abspath(tok_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    model = load_full_model(args.model_path, args.device)

    if args.gt_jsonl_path:
        rows = []
        with open(args.gt_jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        rows = [{"question": r["question"]} for r in ds]
    rows = rows[: args.num_prompts]

    print(f"[probe-collect] model={args.model_path}  {len(rows)} prompts  "
          f"gen={args.gen_length} block={args.block_length} threshold={args.threshold}")

    all_h0, all_arg0, all_y, all_group = [], [], [], []
    t0 = time.time()
    for i, row in enumerate(rows):
        messages = [{"role": "user", "content": GSM8K_USER_TEMPLATE.format(question=row["question"])}]
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt",
        ).to(args.device)
        res = collect_for_prompt(model, prompt_ids, args.gen_length, args.block_length,
                                 args.threshold, args.max_iters_per_block, args.max_blocks)
        if res is None:
            continue
        h0, arg0, y = res
        all_h0.append(h0); all_arg0.append(arg0); all_y.append(y)
        all_group.append(torch.full((h0.shape[0],), i, dtype=torch.long))
        if i < 3 or (i + 1) % 25 == 0:
            n_so_far = sum(t.shape[0] for t in all_h0)
            print(f"[{i+1}/{len(rows)}] +{h0.shape[0]} pos (total {n_so_far}); "
                  f"this prompt iter0!=converged = {(arg0 != y).float().mean().item():.1%}")

    H0 = torch.cat(all_h0, 0).half()
    ARG0 = torch.cat(all_arg0, 0)
    Y = torch.cat(all_y, 0)
    GROUP = torch.cat(all_group, 0)
    flip_rate = (ARG0 != Y).float().mean().item()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)
    torch.save({
        "h0": H0, "iter0_argmax": ARG0, "y_converged": Y, "group": GROUP,
        "meta": {"model_path": args.model_path, "num_prompts": len(rows),
                 "gen_length": args.gen_length, "block_length": args.block_length,
                 "threshold": args.threshold, "d_model": H0.shape[1], "flip_rate": flip_rate},
    }, args.out_path)
    print(f"[probe-collect] saved {H0.shape[0]} positions (D={H0.shape[1]}) -> {args.out_path}")
    print(f"[probe-collect] iter0!=converged overall = {flip_rate:.1%}  (headroom the probe must recover)")
    print(f"[probe-collect] elapsed {time.time()-t0:.1f}s. Next: probe_fit.py --data {args.out_path}")


if __name__ == "__main__":
    main()
