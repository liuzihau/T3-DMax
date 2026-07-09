# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# LLaDA-2.0 GSM8K decode driver (EAGER, official algorithm): calls the OFFICIAL `model.generate(...)` — the
# reference block-diffusion decode from inclusionAI/LLaDA2.X (per-step quota over the full block_length +
# >threshold overshoot + mid-block EOS early-exit). Heavy forwards are counted by wrapping model.forward
# (generate calls self.forward once per denoise step).
#
# Two loaders, SAME algorithm (dFactory's vendored generate() is AST-verified semantically identical to the
# HF checkpoint's — only mask construction style and line wrapping differ):
#   --model_impl fused  (default): vendored LLaDA2MoeModelLM, fused-MoE kernels + sdpa. ~6x faster/forward;
#       REQUIRES a MERGED checkpoint: python scripts/moe_convertor.py -i ../LLaDA2.0-mini \
#           -o ../LLaDA2.0-mini-moe-merge -m merge   (run from dFactory with VeOmni on PYTHONPATH)
#   --model_impl remote: the checkpoint's own remote code (non-fused per-expert MoE) — the byte-authentic
#       reference; use for spot-check A/Bs, too slow for full sweeps (~230ms vs ~40ms per forward).
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
_T3_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DFACTORY = os.path.join(_T3_ROOT, "dFactory")
if os.path.isdir(_DFACTORY) and _DFACTORY not in sys.path:
    sys.path.insert(0, _DFACTORY)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

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


def load_fused(model_path, device="cuda"):
    """Vendored LLaDA2MoeModelLM with fused-MoE + sdpa (the DMax-eager-speed path). `model_path` MUST be a
    MERGED checkpoint (moe_convertor.py -m merge); its generate() is the verified-identical official decode."""
    from models.llada2_moe.configuration_llada2_moe import LLaDA2MoeConfig
    from models.llada2_moe.modeling_llada2_moe import LLaDA2MoeModelLM

    hcfg = LLaDA2MoeConfig.from_pretrained(model_path, trust_remote_code=True)
    if not str(hcfg.model_type).endswith("_veomni"):
        hcfg.model_type = str(hcfg.model_type) + "_veomni"
    hcfg.moe_implementation = "fused"
    m = LLaDA2MoeModelLM.from_pretrained(
        model_path, config=hcfg, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa")
    return m.eval().to(device)


def main():
    p = argparse.ArgumentParser(description="LLaDA-2.0 GSM8K decode (official generate; fused or remote impl)")
    p.add_argument("--model_path", required=True,
                   help="LLaDA2.0 checkpoint dir: MERGED (e.g. ../LLaDA2.0-mini-moe-merge) for --model_impl "
                        "fused, or the original HF layout for --model_impl remote")
    p.add_argument("--model_impl", choices=["fused", "remote"], default="fused",
                   help="fused = vendored fused-MoE class (fast; needs merged ckpt); remote = the checkpoint's "
                        "own code (authentic reference, ~6x slower/forward)")
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--out_path", required=True)
    p.add_argument("--gen_length", type=int, default=512)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, required=True, help="denoise steps per block (official default 32; e.g. 16/9/6)")
    p.add_argument("--threshold", type=float, default=0.95, help="official >threshold overshoot commit (default 0.95)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gt_jsonl_path", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    mp = os.path.abspath(args.model_path)
    tok = AutoTokenizer.from_pretrained(os.path.abspath(args.tokenizer_path or mp), trust_remote_code=True)
    if args.model_impl == "fused":
        model = load_fused(mp, args.device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            mp, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(args.device).eval()
    eos_id = int(getattr(model.config, "eos_token_id", 156892) or 156892)

    # count denoise forwards: the official generate() calls self.forward per step; an instance attribute
    # shadows the bound method
    counter = {"n": 0}
    orig_forward = model.forward

    def counting_forward(*a, **k):
        counter["n"] += 1
        return orig_forward(*a, **k)
    model.forward = counting_forward

    rows = load_gsm8k_test(limit=args.limit, gt_jsonl_path=args.gt_jsonl_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or ".", exist_ok=True)
    print(f"[gsm8k-llada-official] {len(rows)} ex impl={args.model_impl} gen={args.gen_length} "
          f"block={args.block_length} steps={args.steps} thr={args.threshold} T={args.temperature} "
          f"eos={eos_id} -> {args.out_path}")

    t0 = time.time(); tot_tok = 0; tot_wall = 0.0; tot_fwd = 0
    with open(args.out_path, "w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            msgs = [{"role": "user", "content": GSM8K_USER_TEMPLATE.format(question=row["question"])}]
            pid = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_tensors="pt").to(args.device)
            P = pid.shape[1]
            prev = counter["n"]
            torch.cuda.synchronize(); w0 = time.time()
            with torch.no_grad():
                out = model.generate(inputs=pid, gen_length=args.gen_length, block_length=args.block_length,
                                     steps=args.steps, temperature=args.temperature, threshold=args.threshold,
                                     eos_early_stop=True)
            torch.cuda.synchronize(); wall = time.time() - w0
            nfe = counter["n"] - prev
            # official generate has two return paths: mid-block EOS exit includes the prompt, the normal
            # path does not — strip if present, then cut at the first EOS
            seq = out[0]
            if seq.shape[0] >= P and torch.equal(seq[:P], pid[0]):
                seq = seq[P:]
            eos = (seq == eos_id).nonzero(as_tuple=True)[0]
            if eos.numel() > 0:
                seq = seq[:eos[0]]
            text = tok.decode(seq, skip_special_tokens=True)
            ntok = int(seq.shape[0])
            tot_tok += ntok; tot_wall += wall; tot_fwd += nfe
            fh.write(json.dumps({"answer": text, "question": row["question"], "forwards": int(nfe),
                                 "gen_tokens": ntok, "wall_time": round(wall, 4)},
                                ensure_ascii=False) + "\n")
            fh.flush()
            if i < 3 or (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(rows)}] nfe={nfe} tok={ntok} tok/s={ntok/max(wall,1e-6):.1f} "
                      f"tail={text[-160:]!r}")

    n = max(len(rows), 1); dt = time.time() - t0
    print(f"[gsm8k-llada-official] done {dt:.0f}s. tok/s={tot_tok/max(tot_wall,1e-6):.1f} "
          f"fwd/ex={tot_fwd/n:.1f} tok/ex={tot_tok/n:.1f} wall/ex={tot_wall/n:.2f}s")
    print(f"[gsm8k-llada-official] grade: python {os.path.join(_HERE, 'val_gsm8k.py')} --pred-path {args.out_path}")


if __name__ == "__main__":
    main()
