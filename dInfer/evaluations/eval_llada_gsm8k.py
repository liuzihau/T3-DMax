# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# LLaDA-2.0 GSM8K decode driver (EAGER, heavy-only): the vanilla fixed-steps baseline.
# Per block, `--steps` denoise iterations; each commits the per-step quota of highest-confidence
# masked slots (low_confidence remasking, temperature 0), hard token ids every forward (no soft-embed
# re-feed). Block loop / grid alignment / block-causal attn / early-stop mirror generate_dbet's eager
# heavy-only path so wall-clock and forward counts are comparable with eager DMax/DBet numbers.
#
# Output jsonl (same keys as the sglang driver -> one summarizer works for both):
#   {"answer", "question", "forwards", "gen_tokens", "wall_time"}
# Grade:  python val_gsm8k.py --pred-path <jsonl>

import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DINFER_PYTHON = os.path.abspath(os.path.join(_HERE, "..", "python"))
if _DINFER_PYTHON not in sys.path:
    sys.path.insert(0, _DINFER_PYTHON)

from transformers import AutoTokenizer  # noqa: E402

from dinfer.decoding.generate_dbet import MASK_ID, EOS_ID, PAD_ID  # noqa: E402  (import also puts dFactory on sys.path)
from dinfer.decoding.generate_t3d import build_block_causal_mask  # noqa: E402
from dinfer.decoding.parallel_strategy import get_transfer_index  # noqa: E402
from dinfer.decoding.utils import get_num_transfer_tokens  # noqa: E402

GSM8K_USER_TEMPLATE = "Question: {question}\nLet's think step by step\nAnswer:"


def load_gsm8k_test(limit=None, gt_jsonl_path=None):
    if gt_jsonl_path:
        rows = []
        with open(gt_jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        rows = [{"question": r["question"], "answer": r["answer"]} for r in ds]
    if limit is not None:
        rows = rows[:limit]
    return rows


def load_heavy_only(heavy_path, device="cuda"):
    """The FROZEN DMax heavy alone (fused MoE), no drafter — mirrors load_dbet_model's heavy section."""
    from models.llada2_moe.configuration_llada2_moe import LLaDA2MoeConfig
    from models.llada2_moe.modeling_llada2_moe import LLaDA2MoeModelLM

    heavy_path = os.path.abspath(heavy_path)
    hcfg = LLaDA2MoeConfig.from_pretrained(heavy_path, trust_remote_code=True)
    if not str(hcfg.model_type).endswith("_veomni"):
        hcfg.model_type = str(hcfg.model_type) + "_veomni"
    hcfg.moe_implementation = "fused"
    heavy = LLaDA2MoeModelLM.from_pretrained(
        heavy_path, config=hcfg, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa")
    return heavy.eval().to(device)


@torch.no_grad()
def generate_llada_fixed(model, prompt_ids, gen_length, block_length, steps, early_stop=True):
    """Grid-aligned multi-block vanilla-LLaDA generation. Returns (response_ids [n], forwards); response_ids
    excludes the prompt and is cut at the first EOS. The first block may span the prompt tail (grid
    alignment) — quotas are computed over its masked slots only."""
    device = prompt_ids.device
    P = prompt_ids.shape[1]
    first_block_start = (P // block_length) * block_length
    end_target = P + gen_length
    num_blocks = (end_target - first_block_start + block_length - 1) // block_length
    L = first_block_start + num_blocks * block_length

    x = torch.full((1, L), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids
    forwards = 0
    eos_cut = L

    for b in range(num_blocks):
        bs = first_block_start + b * block_length
        be = bs + block_length
        mask_idx = (x[:, bs:be] == MASK_ID)
        if not bool(mask_idx.any()):
            continue
        quotas = get_num_transfer_tokens(mask_idx, steps)
        attn = build_block_causal_mask(be, block_length, dtype=torch.bfloat16, device=device)
        for i in range(steps):
            mask_idx = (x[:, bs:be] == MASK_ID)
            if not bool(mask_idx.any()):
                break
            logits = model(input_ids=x[:, :be], attention_mask=attn).logits[:, bs:be]
            forwards += 1
            x0, tidx = get_transfer_index(
                logits, 0.0, "low_confidence", mask_idx, x[:, bs:be], quotas[:, i], None)
            blk = x[:, bs:be]
            blk[tidx] = x0[tidx]
        if early_stop:
            resp_lo = max(P, bs)
            seg = x[0, resp_lo:be]
            eos_pos = (seg == EOS_ID).nonzero(as_tuple=True)[0]
            if eos_pos.numel() > 0:
                eos_cut = resp_lo + int(eos_pos[0].item())
                if be < L:
                    x[0, be:] = PAD_ID
                break

    return x[0, P:eos_cut].clone(), forwards


def main():
    p = argparse.ArgumentParser(description="LLaDA-2.0 fixed-steps GSM8K decode (eager heavy-only baseline)")
    p.add_argument("--heavy_path", required=True, help="DMax-Math-16B-moe-merge checkpoint")
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--out_path", required=True)
    p.add_argument("--gen_length", type=int, default=512)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, required=True, help="denoise steps per block (e.g. 16 / 9 / 6)")
    p.add_argument("--no_early_stop", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gt_jsonl_path", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(os.path.abspath(args.tokenizer_path or args.heavy_path),
                                        trust_remote_code=True)
    model = load_heavy_only(args.heavy_path, args.device)
    rows = load_gsm8k_test(limit=args.limit, gt_jsonl_path=args.gt_jsonl_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or ".", exist_ok=True)
    print(f"[gsm8k-llada] {len(rows)} ex gen={args.gen_length} block={args.block_length} "
          f"steps={args.steps} -> {args.out_path}")

    t0 = time.time(); tot_tok = 0; tot_wall = 0.0; tot_fwd = 0
    with open(args.out_path, "w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            msgs = [{"role": "user", "content": GSM8K_USER_TEMPLATE.format(question=row["question"])}]
            pid = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_tensors="pt").to(args.device)
            torch.cuda.synchronize(); w0 = time.time()
            ans, nfe = generate_llada_fixed(model, pid, gen_length=args.gen_length,
                                            block_length=args.block_length, steps=args.steps,
                                            early_stop=not args.no_early_stop)
            torch.cuda.synchronize(); wall = time.time() - w0
            text = tok.decode(ans, skip_special_tokens=True)
            ntok = int(ans.shape[0])
            tot_tok += ntok; tot_wall += wall; tot_fwd += nfe
            fh.write(json.dumps({"answer": text, "question": row["question"], "forwards": int(nfe),
                                 "gen_tokens": ntok, "wall_time": round(wall, 4)},
                                ensure_ascii=False) + "\n")
            fh.flush()
            if i < 3 or (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(rows)}] nfe={nfe} tok={ntok} tok/s={ntok/max(wall,1e-6):.1f} "
                      f"tail={text[-160:]!r}")

    n = max(len(rows), 1); dt = time.time() - t0
    print(f"[gsm8k-llada] done {dt:.0f}s. tok/s={tot_tok/max(tot_wall,1e-6):.1f} fwd/ex={tot_fwd/n:.1f} "
          f"tok/ex={tot_tok/n:.1f} wall/ex={tot_wall/n:.2f}s")
    print(f"[gsm8k-llada] grade: python {os.path.join(_HERE, 'val_gsm8k.py')} --pred-path {args.out_path}")


if __name__ == "__main__":
    main()
