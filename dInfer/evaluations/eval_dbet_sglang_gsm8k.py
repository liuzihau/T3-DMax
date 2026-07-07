# Copyright 2026 University of Sydney. Apache-2.0.
"""G2.3 — GSM8K eval for SGLang-DBet (heavy in SGLang + eager drafter). RUN IN THE SGLANG ENV.

  python evaluations/eval_dbet_sglang_gsm8k.py --heavy_path <ORIGINAL per-expert DMax-Math> \
      --drafter_path <hf_ckpt> --out_path preds_sglang.jsonl --limit 100 --gen_length 512 \
      --heavy_threshold 0.9 --draft_threshold 0.9 --draft_top_k 2
  python evaluations/val_gsm8k.py --pred-path preds_sglang.jsonl

This validates ACCURACY parity vs eager DBet (eval_dbet_gsm8k) at the same config — NOT speed (this is still
batch=1; the SGLang win is batching, a later step). `--heavy_path` = the per-expert DMax-Math, used for BOTH the
sglang heavy and the drafter's frozen embed/lm_head/norm (same math model; the dense weights match the merged ckpt).
"""
import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "python")))
from dinfer.decoding.generate_dbet import load_drafter_standalone                 # noqa: E402
from dinfer.decoding.generate_dbet_sglang import generate_dbet_sglang             # noqa: E402
from dinfer.decoding.dbet_sglang_features import HeavyFeatureTap                  # noqa: E402
from eval_dbet_gsm8k import load_gsm8k_test, GSM8K_USER_TEMPLATE                  # noqa: E402
from transformers import AutoTokenizer                                           # noqa: E402


def build_sglang(heavy_path, sel_layers=(1, 10, 19), max_length=2048, master_port="23520"):
    """Bring up the SGLang diffusion heavy (tp=1, cuda graphs OFF so the tap fires) + a HeavyFeatureTap.
    Mirrors evaluations/validate_sglang_hsel.py / eval_dinfer_sglang.py; returns (runner, tap, device)."""
    from transformers import AutoConfig
    from sglang.srt.server_args import ServerArgs
    from sglang.srt import distributed
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.layers.moe import initialize_moe_config
    from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
    from dinfer.decoding.diffusion_runner import ModelRunner

    device = torch.device("cuda:0"); torch.cuda.set_device(0)
    os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", master_port)
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, 1, 1, backend="nccl")
    model_config = AutoConfig.from_pretrained(heavy_path, trust_remote_code=True)
    server_args = ServerArgs(model_path=heavy_path, enable_dp_attention=True, trust_remote_code=True,
                             tp_size=1, dp_size=1, pp_size=1)
    try:
        from sglang.srt.server_args import set_global_server_args_for_scheduler
        set_global_server_args_for_scheduler(server_args)
    except ImportError:
        pass
    initialize_dp_attention(server_args=server_args, model_config=model_config)
    initialize_moe_config(server_args)
    torch.set_default_dtype(torch.bfloat16)
    model = LLaDA2SGLangLM(config=model_config, expert_map_path=".").eval()
    model.load_weights(heavy_path, device=device)
    model = model.to(device)
    runner = ModelRunner(model, device, enable_cuda_graph=False, server_args=server_args, max_length=max_length)
    sel = list(getattr(model_config, "sel_layers_list", sel_layers))
    tap = HeavyFeatureTap(runner.model.model.layers, sel_layers=sel, num_layers=len(runner.model.model.layers),
                          final_norm=runner.model.model.norm)
    return runner, tap, device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heavy_path", required=True, help="ORIGINAL per-expert DMax-Math (sglang heavy + drafter frozen)")
    p.add_argument("--drafter_path", required=True)
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--out_path", required=True)
    p.add_argument("--gen_length", type=int, default=512)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--heavy_threshold", type=float, default=0.9)
    p.add_argument("--draft_threshold", type=float, default=0.9)
    p.add_argument("--draft_top_k", type=int, default=2)
    p.add_argument("--heavy_tau", type=float, default=1.0)
    p.add_argument("--draft_tau", type=float, default=1.0)
    p.add_argument("--no_draft_fix", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gt_jsonl_path", default=None)
    args = p.parse_args()

    runner, tap, device = build_sglang(args.heavy_path, max_length=args.gen_length + 256)
    draft = load_drafter_standalone(args.drafter_path, args.heavy_path, device=str(device))
    tok = AutoTokenizer.from_pretrained(os.path.abspath(args.tokenizer_path or args.heavy_path), trust_remote_code=True)
    rows = load_gsm8k_test(limit=args.limit, gt_jsonl_path=args.gt_jsonl_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or ".", exist_ok=True)
    print(f"[sglang-dbet] {len(rows)} ex gen={args.gen_length} h{args.heavy_threshold} d{args.draft_threshold} "
          f"k{args.draft_top_k} fix={not args.no_draft_fix} -> {args.out_path}")

    t0 = time.time(); tot_h = tot_d = tot_tok = 0; tot_wall = 0.0
    with open(args.out_path, "w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            msgs = [{"role": "user", "content": GSM8K_USER_TEMPLATE.format(question=row["question"])}]
            pid = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_tensors="pt").to(device)
            resp, stats = generate_dbet_sglang(
                runner, tap, draft, pid, gen_length=args.gen_length, block_length=args.block_length,
                heavy_threshold=args.heavy_threshold, draft_threshold=args.draft_threshold,
                heavy_tau=args.heavy_tau, draft_tau=args.draft_tau, draft_top_k=args.draft_top_k,
                draft_fix=not args.no_draft_fix)
            text = tok.decode(resp, skip_special_tokens=True)
            tot_h += stats.heavy_forwards; tot_d += stats.draft_forwards
            tot_wall += stats.wall_time; tot_tok += int(resp.shape[0])
            fh.write(json.dumps({
                "answer": text, "question": row["question"],
                "heavy_forwards": stats.heavy_forwards, "draft_forwards": stats.draft_forwards,
                "heavy_commits": stats.heavy_commits, "draft_commits": stats.draft_commits,
                "draft_fixes": stats.draft_fixes, "wall_time": round(stats.wall_time, 4),
                "gen_tokens": int(resp.shape[0]),
            }, ensure_ascii=False) + "\n")
            fh.flush()
            if i < 3 or (i + 1) % 25 == 0:
                print(f"[{i+1}/{len(rows)}] heavy={stats.heavy_forwards} draft={stats.draft_forwards} "
                      f"draft_commits={stats.draft_commits} tail={text[-160:]!r}")

    n = max(len(rows), 1); dt = time.time() - t0
    print(f"[sglang-dbet] done {dt:.0f}s. mean heavy/ex={tot_h/n:.1f} draft/ex={tot_d/n:.1f} "
          f"wall/ex={tot_wall/n:.2f}s tok/s={tot_tok/max(tot_wall,1e-6):.1f} (batch=1; batching is a later step)")
    print(f"[sglang-dbet] grade: python {os.path.join(_HERE,'val_gsm8k.py')} --pred-path {args.out_path}"
          + (f" --limit {args.limit}" if args.limit else ""))


if __name__ == "__main__":
    main()
