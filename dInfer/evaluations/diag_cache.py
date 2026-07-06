# Copyright 2026 University of Sydney. Apache-2.0.
"""Localize the DBet prefix-KV cache non-exactness AND prove it's float precision. For ONE prompt, compare
block-0's FIRST heavy forward (no decode loop, no finalization) cache-partial vs no-cache-full:
  - block logits / block h_sel  (partial forward)     - prefix h_sel  (prompt cache)
Run at different --dtype (+ --exact_moe to remove the fused-kernel batch-dependence):
  bf16  -> max|Δ| ~ 1e-1 (non-associative fused-MoE + flash-attn tiling)
  fp32 --exact_moe -> max|Δ| ~ 1e-4  => the divergence is FLOAT PRECISION (summation order), not the math.

  python evaluations/diag_cache.py --drafter_path <ckpt> --heavy_path <DMax> --dtype float32 --exact_moe
"""
import argparse
import os
import sys
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "python")))
from dinfer.decoding.generate_dbet import (build_block_causal_mask, load_dbet_model,     # noqa: E402
                                           _new_dynamic_cache, MASK_ID)
from eval_dbet_gsm8k import load_gsm8k_test                                              # noqa: E402
from transformers import AutoTokenizer                                                   # noqa: E402


def _patch_exact_moe(model):
    def _exact(self, hidden_states, expert_idx=None, routing_weights=None, selected_experts=None):
        if expert_idx is not None:
            g = torch.matmul(hidden_states, self.gate_proj[expert_idx].transpose(0, 1))
            u = torch.matmul(hidden_states, self.up_proj[expert_idx].transpose(0, 1))
            return torch.matmul(self.act_fn(g) * u, self.down_proj[expert_idx].transpose(0, 1))
        out = torch.zeros_like(hidden_states)
        for k in range(selected_experts.shape[1]):
            eids = selected_experts[:, k]; w = routing_weights[:, k]
            for e in torch.unique(eids).tolist():
                m = eids == e; hs = hidden_states[m]
                g = torch.matmul(hs, self.gate_proj[e].transpose(0, 1))
                u = torch.matmul(hs, self.up_proj[e].transpose(0, 1))
                y = torch.matmul(self.act_fn(g) * u, self.down_proj[e].transpose(0, 1))
                out[m] += w[m].unsqueeze(-1) * y
        return out
    n = 0
    for mod in model.modules():
        if all(hasattr(mod, a) for a in ("gate_proj", "up_proj", "down_proj", "num_experts", "act_fn")) \
           and mod.gate_proj.dim() == 3:
            mod.forward = types.MethodType(_exact, mod); n += 1
    print(f"[exact_moe] patched {n} fused-experts modules")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drafter_path", required=True)
    p.add_argument("--heavy_path", required=True)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--gen_length", type=int, default=512)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--exact_moe", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    DT = getattr(torch, args.dtype)

    tok = AutoTokenizer.from_pretrained(os.path.abspath(args.tokenizer_path or args.heavy_path), trust_remote_code=True)
    model = load_dbet_model(args.drafter_path, args.heavy_path, args.device)
    model.to(DT)
    if args.exact_moe:
        _patch_exact_moe(model)
    row = load_gsm8k_test(limit=1)[0]
    pid = tok.apply_chat_template([{"role": "user", "content": row["question"]}],
                                  add_generation_prompt=True, tokenize=True, return_tensors="pt").to(args.device)
    P = pid.shape[1]; block = args.block_length
    fbs = (P // block) * block; bs, be = fbs, fbs + block
    x = torch.full((1, be), MASK_ID, dtype=torch.long, device=args.device); x[:, :P] = pid
    embed = model.draft.frozen_embed
    print(f"P={P} bs={bs} be={be} dtype={args.dtype} exact_moe={args.exact_moe}")

    # NO-CACHE full [0,be)
    m_full = build_block_causal_mask(be, block, dtype=DT, device=args.device)
    sig_f = model.extract_heavy_signals(x[:, :be], attention_mask=m_full, inputs_embeds=embed(x[:, :be]))
    logits_f, hblk_f, hpre_f = sig_f["logits"][:, bs:be], sig_f["h_sel"][:, bs:be], sig_f["h_sel"][:, :bs]

    # CACHE: prefix [0,bs) then partial block  (masks in DT, inline so no hard-coded bf16)
    cache = _new_dynamic_cache()
    m_pre = build_block_causal_mask(bs, block, dtype=DT, device=args.device)
    sig_pre = model.extract_heavy_signals(x[:, :bs], attention_mask=m_pre, inputs_embeds=embed(x[:, :bs]),
                                          past_key_values=cache, use_cache=True)
    prefix_hsel, cache = sig_pre["h_sel"], sig_pre["past_key_values"]
    m_blk = torch.zeros(1, 1, be - bs, be, dtype=DT, device=args.device)
    sig_c = model.extract_heavy_signals(x[:, bs:be], attention_mask=m_blk, inputs_embeds=embed(x[:, bs:be]),
                                        past_key_values=cache, use_cache=True)
    logits_c, hblk_c = sig_c["logits"], sig_c["h_sel"]

    def rep(name, a, b):
        d = (a.float() - b.float()).abs()
        am = (a.argmax(-1) == b.argmax(-1)).float().mean().item() if a.dim() == 3 else float("nan")
        print(f"  {name:14s} max|Δ|={d.max().item():.6f}  mean|Δ|={d.mean().item():.7f}  argmax_match={am:.4f}")

    print("block-0 first forward, cache vs no-cache:")
    rep("block logits", logits_c, logits_f)
    rep("block h_sel", hblk_c, hblk_f)
    rep("prefix h_sel", prefix_hsel, hpre_f)
    print("\nfp32 --exact_moe collapsing these to ~1e-4 => pure float summation-order (bf16), not a logic bug.")


if __name__ == "__main__":
    main()
