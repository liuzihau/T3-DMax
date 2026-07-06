# Copyright 2026 University of Sydney. Apache-2.0.
"""Acceptance gate for the DBet prefix-KV cache (--use_cache). Runs each GSM8K prompt BOTH ways
(no-cache vs cache) in one process and asserts the generated token ids are IDENTICAL. The cache is a
pure speed optimization, so any mismatch = a cache bug (crop timing / finalization / mask / position).

  python evaluations/validate_cache.py --drafter_path <hf_ckpt> --heavy_path <DMax> --limit 10 \
    --gen_length 512 --block_length 32 --heavy_threshold 0.9 --draft_threshold 0.9 --draft_top_k 2
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "python")))
from dinfer.decoding.generate_dbet import generate_dbet, load_dbet_model          # noqa: E402
from eval_dbet_gsm8k import load_gsm8k_test                                        # noqa: E402
from transformers import AutoTokenizer                                            # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drafter_path", required=True)
    p.add_argument("--heavy_path", required=True)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--gen_length", type=int, default=512)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--heavy_threshold", type=float, default=0.9)
    p.add_argument("--draft_threshold", type=float, default=0.9)
    p.add_argument("--heavy_top_k", type=int, default=1)
    p.add_argument("--draft_top_k", type=int, default=2)
    p.add_argument("--draft_committed_soft", action="store_true")
    p.add_argument("--no_draft_fix", action="store_true")
    p.add_argument("--gt_jsonl_path", default=None)
    p.add_argument("--exact_moe", action="store_true",
                   help="force the non-fused (row-independent, length-invariant) MoE path -> isolates the cache "
                        "LOGIC from the batch-dependent veomni fused kernel. Expect 10/10 if the cache is correct.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="cast the model; fp32 kills bf16 non-associativity (tokens should match).")
    p.add_argument("--math_attn", action="store_true",
                   help="force SDPA MATH backend (full score matrix, no flash tiling) -> attention becomes "
                        "shape-independent. With --exact_moe this removes BOTH non-associativity sources -> "
                        "expect 10/10 BIT-IDENTICAL even in bf16 (the definitive cache-correctness proof).")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(os.path.abspath(args.tokenizer_path or args.heavy_path), trust_remote_code=True)
    model = load_dbet_model(args.drafter_path, args.heavy_path, args.device)
    model.to(getattr(torch, args.dtype))
    if args.exact_moe:
        import types

        def _exact_experts(self, hidden_states, expert_idx=None, routing_weights=None, selected_experts=None):
            # row-independent (length-invariant) replacement for the fused kernel: per-token, per-expert eager
            # matmuls, matching the module's own eager path (lines 298-302 of modeling_llada2_moe).
            if expert_idx is not None:
                g = torch.matmul(hidden_states, self.gate_proj[expert_idx].transpose(0, 1))
                u = torch.matmul(hidden_states, self.up_proj[expert_idx].transpose(0, 1))
                return torch.matmul(self.act_fn(g) * u, self.down_proj[expert_idx].transpose(0, 1))
            out = torch.zeros_like(hidden_states)
            for k in range(selected_experts.shape[1]):
                eids = selected_experts[:, k]; w = routing_weights[:, k]
                for e in torch.unique(eids).tolist():
                    m = eids == e
                    hs = hidden_states[m]
                    g = torch.matmul(hs, self.gate_proj[e].transpose(0, 1))
                    u = torch.matmul(hs, self.up_proj[e].transpose(0, 1))
                    y = torch.matmul(self.act_fn(g) * u, self.down_proj[e].transpose(0, 1))
                    out[m] += w[m].unsqueeze(-1) * y
            return out

        n = 0
        for mod in model.modules():
            if all(hasattr(mod, a) for a in ("gate_proj", "up_proj", "down_proj", "num_experts", "act_fn")) \
               and getattr(mod, "gate_proj").dim() == 3:            # the fused EXPERTS module (3D stacked weights)
                mod.forward = types.MethodType(_exact_experts, mod); n += 1
        print(f"[exact_moe] patched {n} fused-experts modules to the row-independent exact path (length-invariant)")
    rows = load_gsm8k_test(limit=args.limit, gt_jsonl_path=args.gt_jsonl_path)

    import contextlib
    if args.math_attn:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        attn_ctx = lambda: sdpa_kernel(SDPBackend.MATH)
    else:
        attn_ctx = contextlib.nullcontext

    def gen(prompt_ids, use_cache):
        with attn_ctx():
            r, s = generate_dbet(
                model, prompt_ids, gen_length=args.gen_length, block_length=args.block_length,
                heavy_threshold=args.heavy_threshold, draft_threshold=args.draft_threshold,
                heavy_top_k=args.heavy_top_k, draft_top_k=args.draft_top_k,
                draft_committed_soft=args.draft_committed_soft, draft_fix=not args.no_draft_fix,
                use_cache=use_cache)
        return r, s

    n_ok = 0
    for i, row in enumerate(rows):
        msgs = [{"role": "user", "content": row["question"]}]
        pid = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt").to(args.device)
        r0, s0 = gen(pid, use_cache=False)
        r1, s1 = gen(pid, use_cache=True)
        same = (r0.shape == r1.shape) and bool(torch.equal(r0, r1))
        if same:
            n_ok += 1
            tag = "OK "
        else:
            # first divergence
            m = min(r0.shape[0], r1.shape[0])
            div = next((k for k in range(m) if int(r0[k]) != int(r1[k])), m)
            tag = f"MISMATCH @tok{div} (len {r0.shape[0]} vs {r1.shape[0]})"
        print(f"[{i+1}/{len(rows)}] {tag}  | heavy_fwd nocache={s0.heavy_forwards} cache={s1.heavy_forwards}")

    print(f"\n=== {n_ok}/{len(rows)} identical. "
          f"{'PASS — cache is correct, trust the timing.' if n_ok == len(rows) else 'FAIL — cache bug, do NOT trust timing.'} ===")


if __name__ == "__main__":
    main()
