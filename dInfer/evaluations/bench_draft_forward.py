# Copyright 2026 University of Sydney. Apache-2.0.
"""Phase-3 KILL-SWITCH: time the DBet DRAFT forward (eager + torch.compile) against the optimized heavy's ~5.6ms.

The whole question for DBet-under-the-optimized-stack is draft_time vs heavy_time: DBet spends one draft forward
per heavy forward to commit extra tokens (cutting the # of expensive heavy forwards). Win iff the draft is cheap.
No sglang / no model-file surgery needed here — just the standalone drafter on synthetic inputs of realistic shapes.

  python evaluations/bench_draft_forward.py --heavy_path <DMax-Math> --drafter_path <hf_ckpt> [--compile]

READ the ms/forward: <~1-2ms (compiled) => DBet has room under the stack -> do the graph-safe capture surgery;
~4-5ms => the draft is as costly as the heavy -> DBet won't help at batch=1, STOP.
"""
import argparse
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "python")))
from dinfer.decoding.generate_dbet import load_drafter_standalone                 # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heavy_path", required=True, help="for the frozen embed/lm_head/norm")
    p.add_argument("--drafter_path", required=True)
    p.add_argument("--block", type=int, default=32)
    p.add_argument("--prefix", type=int, default=256, help="settled-prefix length (mid-decode)")
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--compile", action="store_true", help="torch.compile the drafter (the optimized number)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    draft = load_drafter_standalone(args.drafter_path, args.heavy_path, device=args.device)
    cfg = draft.config
    dev = args.device
    V, D = draft.frozen_lm_head.weight.shape                          # lm_head [V, D]
    mD = int(getattr(cfg, "m", 3)) * D                               # h_sel width = |sel_layers| * hidden
    blk, P = args.block, args.prefix

    ids = torch.randint(0, V, (1, blk), device=dev)
    hlog = torch.randn(1, blk, V, device=dev, dtype=torch.bfloat16)
    hsel = torch.randn(1, blk, mD, device=dev, dtype=torch.bfloat16)
    hlast = torch.randn(1, blk, D, device=dev, dtype=torch.bfloat16)
    hpre = torch.randn(1, P, mD, device=dev, dtype=torch.bfloat16)

    fn = torch.compile(draft) if args.compile else draft

    def one():
        return fn(input_ids=ids, heavy_logits=hlog, h_sel_denoise=hsel, h_last_denoise=hlast,
                  h_sel_prefix=hpre, attention_mask=None, position_ids=None, denoise_mask=None, tau=None)

    with torch.no_grad():
        for _ in range(15):                                          # warmup (incl. torch.compile trace)
            one()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            one()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / args.iters * 1e3

    print(f"\n[draft-bench] block={blk} prefix={P} compile={args.compile} -> {ms:.2f} ms/draft-forward")
    print(f"  heavy (optimized, batch=1) ~5.6 ms/forward  =>  draft is {5.6/ms:.1f}x cheaper")
    print("  VERDICT: draft << heavy (>~3x) => DBet has room under the stack, do the graph capture; "
          "draft ~ heavy => STOP (no batch=1 win).")


if __name__ == "__main__":
    main()
