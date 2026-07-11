# Copyright 2026 University of Sydney. Apache-2.0.
"""Phase-3 DRIVER: DBet under DMax's OPTIMIZED cached + CUDA-graph decode (DbetBlockDiffusionLLM) on GSM8K.
Reports accuracy (grade after) + tok/s vs the ~927 heavy-only baseline.

  # 1) DEBUG the injection first -- graphs OFF, small N (verify draft_commits>0 and accuracy ~= G2 89%):
  python evaluations/eval_dbet_sglang_cached_gsm8k.py --heavy_path <DMax-Math per-expert> \
      --drafter_path <hf_ckpt> --out_path preds_cached.jsonl --limit 20
  # 2) SPEED -- graphs ON + compiled draft:
  python evaluations/eval_dbet_sglang_cached_gsm8k.py --heavy_path ... --drafter_path ... \
      --out_path preds_cached.jsonl --limit 100 --cuda_graph --compile_draft
  # then: python evaluations/val_gsm8k.py --pred-path preds_cached.jsonl

The heavy runs through DMax's cached ModelRunner (prefix KV + graphs); the graph-safe buffer tap
(modeling_llada2_moe_sglang_dbet) feeds the drafter injected at forward_uniform. See generate_dbet_sglang_cached.py.
"""
import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "python")))
sys.path.insert(0, _HERE)
from dinfer.decoding.generate_dbet import load_drafter_standalone                       # noqa: E402
from dinfer.decoding.generate_dbet_sglang_cached import DbetBlockDiffusionLLM           # noqa: E402
from eval_dbet_gsm8k import load_gsm8k_test, GSM8K_USER_TEMPLATE                         # noqa: E402
from transformers import AutoTokenizer, AutoConfig                                      # noqa: E402

MASK_ID, EOS_ID = 156895, 156892


def build(heavy_path, sel_default, max_length, use_cuda_graph, enable_tap=True, master_port="23521"):
    """Bring up the sglang heavy from the TAP COPY (modeling_llada2_moe_sglang_dbet) + a cached ModelRunner."""
    from sglang.srt.server_args import ServerArgs
    from sglang.srt import distributed
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.layers.moe import initialize_moe_config
    from dinfer.model.modeling_llada2_moe_sglang_dbet import LLaDA2SGLangLM              # the COPY w/ enable_dbet_tap
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
    sel = list(getattr(model_config, "sel_layers_list", sel_default))
    # CRITICAL: enable the buffer tap BEFORE the ModelRunner captures the CUDA graph in its ctor -- else the graph
    # has no copy_ ops and the buffer is never refreshed on replay (drafter reads stale h_sel). Skipped for --no_draft
    # (pure heavy-only baseline: LLaDA/DMax through the identical decode, no drafter, no tap).
    if enable_tap:
        model.model.enable_dbet_tap(sel, max_bs=1, max_len=256)
    runner = ModelRunner(model, device, enable_cuda_graph=use_cuda_graph, server_args=server_args,
                         max_length=max_length)
    return runner, sel, server_args, device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heavy_path", required=True, help="per-expert sglang heavy (DMax-Math / LLaDA2); also frozen drafter pieces")
    p.add_argument("--drafter_path", default=None, help="required unless --no_draft (heavy-only baseline)")
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--out_path", required=True)
    p.add_argument("--gen_length", type=int, default=512)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--heavy_threshold", type=float, default=0.9)
    p.add_argument("--decoder", choices=["threshold", "fixed"], default="threshold",
                   help="'threshold' = DMax ThresholdParallelDecoder; 'fixed' = vanilla-LLaDA fixed-steps "
                        "(--steps per block, hard-token feed, requires --no_draft).")
    p.add_argument("--steps", type=int, default=16, help="per-block denoise steps for --decoder fixed")
    p.add_argument("--draft_threshold", type=float, default=0.9)
    p.add_argument("--draft_fix_threshold", type=float, default=None,
                   help="separate conf gate for FIX (override committed slots); default = --draft_threshold. "
                        "Golden diag says FIX is only net-positive at >=0.9 while EXTEND can run lower.")
    p.add_argument("--no_markov", action="store_true",
                   help="ABLATION: drop a loaded markov head (semi-AR walk off).")
    p.add_argument("--draft_refine", action="store_true",
                   help="also run the drafter during the REFINE/CONVERGE rounds (block fully committed, heavy "
                        "still verifying) — FIX-only there; targets the ~40-fwd verification floor. Same flag "
                        "exists on the eager driver (default off in BOTH since 0711).")
    p.add_argument("--draft_committed_mix", type=str, default=None,
                   help="committed-slot input mode for the drafter (applies to the real decode AND --diag): "
                        "'conf' = DMax confidence-weighted p*E(tok)+(1-p)*E(MASK) renormalized; a float = fixed "
                        "H/M interpolation (1.0 hard, 0.0 MASK); default None = hard (route H, v1-comparable).")
    p.add_argument("--draft_top_k", type=int, default=2)
    p.add_argument("--draft_tau", type=float, default=1.0,
                   help="temperature of the soft embed feeding drafter COMMITS to the next heavy forward. "
                        "(Since 0711 the drafter's own input conditioning uses the train-matched 0.8, "
                        "matching the eager path; before, this flag did double duty at 1.0.)")
    p.add_argument("--no_draft_fix", action="store_true")
    p.add_argument("--diag", action="store_true", help="FIX-quality DRY-RUN diagnostic: heavy decodes as pure DMax "
                   "(draft never commits); dry-run the draft each heavy run and compare to the NEXT heavy run's argmax; "
                   "sweeps the conf threshold -> precision/recall/token-acc. Run WITHOUT --no_draft.")
    p.add_argument("--no_draft", action="store_true", help="pure heavy-only baseline in the SAME cached decode "
                   "(threshold can't disable the drafter: EXTEND always commits the leftmost slot for progress)")
    p.add_argument("--cuda_graph", action="store_true", help="enable CUDA graphs (default OFF -> debug the injection)")
    p.add_argument("--compile_draft", action="store_true", help="torch.compile the drafter (the optimized draft time)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gt_jsonl_path", default=None)
    args = p.parse_args()

    from dinfer import BlockIteratorFactory, KVCacheFactory, ThresholdParallelDecoder
    use_draft = not args.no_draft
    max_length = args.gen_length + 256
    runner, sel, server_args, device = build(args.heavy_path, (1, 10, 19), max_length, args.cuda_graph,
                                             enable_tap=use_draft)
    if use_draft:
        if not args.drafter_path:
            raise SystemExit("--drafter_path is required unless --no_draft")
        draft = load_drafter_standalone(args.drafter_path, args.heavy_path, device=str(device))
        if args.no_markov and getattr(draft, "markov_head", None) is not None:
            draft.markov_head = None
            print("[sglang-dbet-cached] --no_markov: markov head DROPPED (ablation; no semi-AR walk)")
        if args.compile_draft:
            draft = torch.compile(draft)
    else:
        draft = None

    if args.decoder == "fixed":
        if not args.no_draft:
            raise SystemExit("--decoder fixed is the vanilla-LLaDA heavy baseline; pass --no_draft")
        from dinfer.decoding.parallel_strategy import FixedParallelDecoder
        decoder = FixedParallelDecoder(temperature=0, steps=args.steps, mask_id=MASK_ID, eos_id=EOS_ID)
    else:
        decoder = ThresholdParallelDecoder(temperature=0, threshold=args.heavy_threshold, mask_id=MASK_ID, eos_id=EOS_ID)
    cache_factory = KVCacheFactory("prefix", is_bd_model=True, backend="sglang", max_length=max_length)
    dllm = DbetBlockDiffusionLLM(
        runner, decoder, BlockIteratorFactory(start_block_align=True, use_block_diffusion=True),
        cache_factory=cache_factory, draft=draft, sel_layers=sel,
        draft_threshold=args.draft_threshold, draft_tau=args.draft_tau, draft_top_k=args.draft_top_k,
        draft_fix=not args.no_draft_fix, draft_enabled=not args.no_draft,
        draft_fix_threshold=args.draft_fix_threshold, draft_committed_mix=args.draft_committed_mix,
        draft_refine=args.draft_refine,
        early_stop=True, maximum_unroll=4, expected_tpf=4, backend="sglang")
    if args.diag:
        dllm.diff_iteration.diag_on = True

    tok = AutoTokenizer.from_pretrained(os.path.abspath(args.tokenizer_path or args.heavy_path), trust_remote_code=True)
    rows = load_gsm8k_test(limit=args.limit, gt_jsonl_path=args.gt_jsonl_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or ".", exist_ok=True)
    dec_tag = f"fixed_s{args.steps}" if args.decoder == "fixed" else f"h{args.heavy_threshold}"
    print(f"[sglang-dbet-cached] {len(rows)} ex gen={args.gen_length} block={args.block_length} "
          f"graphs={args.cuda_graph} compile_draft={args.compile_draft} dec={dec_tag} "
          f"d{args.draft_threshold} dfix{args.draft_fix_threshold if args.draft_fix_threshold is not None else args.draft_threshold} "
          f"mix={args.draft_committed_mix or 'hard'} "
          f"k{args.draft_top_k} fix={not args.no_draft_fix} -> {args.out_path}")

    if args.compile_draft and rows:                                   # warm up the compiled draft OFF the timer
        wmsg = [{"role": "user", "content": GSM8K_USER_TEMPLATE.format(question=rows[0]["question"])}]
        wpid = tok.apply_chat_template(wmsg, add_generation_prompt=True, tokenize=True, return_tensors="pt").to(device)
        for _ in range(2):
            dllm.diff_iteration.reset()
            dllm.generate(wpid, gen_length=args.gen_length, block_length=args.block_length)
        torch.cuda.synchronize()
        print("[warmup] compiled draft warmed (2 generates) -> timed loop is steady-state")

    t0 = time.time(); tot_tok = 0; tot_wall = 0.0; tot_fwd = 0; tot_dc = 0
    with open(args.out_path, "w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            msgs = [{"role": "user", "content": GSM8K_USER_TEMPLATE.format(question=row["question"])}]
            pid = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_tensors="pt").to(device)
            dllm.diff_iteration.reset()                                   # fresh cross-block draft cache per example
            prev_fwd = dllm.num_forwards
            torch.cuda.synchronize(); w0 = time.time()
            out = dllm.generate(pid, gen_length=args.gen_length, block_length=args.block_length)
            torch.cuda.synchronize(); wall = time.time() - w0
            nfe = dllm.num_forwards - prev_fwd
            dc = dllm.diff_iteration.draft_commits
            ans = out[0, pid.shape[1]:]
            eos = (ans == EOS_ID).nonzero(as_tuple=True)[0]
            if eos.numel() > 0:
                ans = ans[:eos[0]]
            text = tok.decode(ans, skip_special_tokens=True)
            ntok = int(ans.shape[0])
            tot_tok += ntok; tot_wall += wall; tot_fwd += nfe; tot_dc += dc
            fh.write(json.dumps({"answer": text, "question": row["question"], "forwards": int(nfe),
                                 "draft_commits": int(dc), "gen_tokens": ntok,
                                 "wall_time": round(wall, 4)}, ensure_ascii=False) + "\n")
            fh.flush()
            if i < 3 or (i + 1) % 25 == 0:
                print(f"[{i+1}/{len(rows)}] nfe={nfe} draft_commits={dc} tok={ntok} "
                      f"tok/s={ntok/max(wall,1e-6):.1f} tail={text[-160:]!r}")

    n = max(len(rows), 1); dt = time.time() - t0
    print(f"[sglang-dbet-cached] done {dt:.0f}s. tok/s={tot_tok/max(tot_wall,1e-6):.1f} fwd/ex={tot_fwd/n:.1f} "
          f"draft_commits/ex={tot_dc/n:.1f}  (heavy-only baseline ~927 tok/s; draft_commits=0 => injection NOT hit)")
    print(f"[sglang-dbet-cached] grade: python {os.path.join(_HERE,'val_gsm8k.py')} --pred-path {args.out_path}"
          + (f" --limit {args.limit}" if args.limit else ""))

    if args.diag:
        dllm.diff_iteration.diag_dry_report()


if __name__ == "__main__":
    main()
