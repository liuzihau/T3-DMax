#!/usr/bin/env bash
# Copyright 2026 University of Sydney. Apache-2.0.
#
# CEILING CHECK (plan Phase 0, revised): heavy-only SGLang tps at BATCH=1 with the FULL optimized stack
# (CUDA graphs + torch.compile + prefix KV cache) on GSM8K — DMax's own path (eval_dinfer_sglang.py, show_speed).
# Goal: confirm the speed lever is the OPTIMIZED forward, NOT batching. Compare the reported `tps=` to:
#   - our EAGER heavy (~120-180 tok/s, eval_dbet_gsm8k --heavy_only),
#   - our G2 sglang-DBet (CUDA graphs OFF, still slow),
#   - DMax's ~1000 tok/s claim.
# If this hits ~1000 at batch=1 -> the optimization (graphs/compile/cache) is the win, and DBet needs it (Phase 3:
# graph-safe feature capture), not batching.
#
# RUN IN THE SGLANG ENV, single GPU:
#   MODEL=./DMax-Math-16B bash evaluations/bench_sglang_heavy_tps.sh
# Env overrides: MODEL, LEN (gen_length 512), THR (threshold 0.9 = match our DBet; 0.5 = DMax default/faster), LIMIT (20).
set -u
cd "$(dirname "$0")"                                        # -> dInfer/evaluations
export PYTHONPATH="$(pwd)/../python:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1 HF_DATASETS_TRUST_REMOTE_CODE=1 TRANSFORMERS_TRUST_REMOTE_CODE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${MODEL:-./DMax-Math-16B}"                           # ORIGINAL per-expert DMax-Math (sglang loader format)
LEN="${LEN:-512}"                                           # gen_length (match our DBet; tps is ~rate-invariant)
THR="${THR:-0.9}"                                           # threshold (0.9 = our DBet config; 0.5 = DMax default)
LIMIT="${LIMIT:-20}"                                        # #examples (quick; tps is a rate so small is fine)
PORT="${PORT:-23530}"

echo "[bench] heavy-only sglang tps: model=$MODEL gen=$LEN thr=$THR limit=$LIMIT (batch=1, tp=1, graphs+compile+prefix-cache)"
echo "[bench] READ the 'tps=' in the [iter ...] lines (mean in parens) — that is the batch=1 ceiling."

python eval_dinfer_sglang.py --tasks gsm8k_llada_mini \
  --confirm_run_unsafe_code --model dInfer_eval --limit "$LIMIT" \
  --model_args model_path="$MODEL",gen_length="$LEN",block_length=32,threshold="$THR",low_threshold=0.0,show_speed=True,save_dir=./bench_heavy,parallel_decoding=threshold,cache=prefix,warmup_times=0,use_compile=True,tp_size=1,parallel=tp,cont_weight=0,use_credit=False,prefix_look=0,after_look=0,gpus="0",model_type=llada2,use_bd=True,master_port="$PORT",save_samples=False \
  --output_path ./bench_heavy --include_path tasks --apply_chat_template
