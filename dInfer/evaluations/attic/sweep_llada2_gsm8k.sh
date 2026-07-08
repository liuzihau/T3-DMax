#!/usr/bin/env bash
# Copyright 2026 University of Sydney. Apache-2.0.
#
# LLaDA-2.0 (BASE model) GSM8K BASELINE sweep — the ORIGINAL dInfer decode (eval_dinfer_sglang.py +
# lm-eval-harness, sglang/TP fast path), NOT our DBet decode. Sweeps threshold {0.5,0.7,0.9} at
# gen_length 512 to line up with our DBet report. Reports lm-eval GSM8K accuracy + show_speed throughput.
# This is the pre-DMax base model (DMax = its math fine-tune) — a reference we don't otherwise have.
# Based on eval_llada_mini.sh / eval_llada_dmax_math.sh.
#   MODEL_PATH=/abs/LLaDA-2.0-xxx-moe-merge bash evaluations/sweep_llada2_gsm8k.sh
#   THRESHOLDS="0.5 0.7 0.9" TP=2 GPUS='0;1' CUDA_VISIBLE_DEVICES=0,1 LENGTH=512 bash evaluations/sweep_llada2_gsm8k.sh
#
# NOTE: set MODEL_PATH to the LLaDA-2.0 base that DMax was fine-tuned from (default below is the mini —
# change it if you want the 16B base). Graded by lm-eval-harness (not our val_gsm8k.py).

set -u
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # .../dInfer/evaluations
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}/../python:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

MODEL_PATH="${MODEL_PATH:-../../LLaDA2.0-mini-moe-merge}"      # <-- set to the LLaDA-2.0 base you want to baseline
MODEL_TYPE="${MODEL_TYPE:-llada2}"
LENGTH="${LENGTH:-512}"                                       # aligned with the DBet report
BLOCK="${BLOCK:-32}"
THRESHOLDS="${THRESHOLDS:-0.5 0.7 0.9}"
TASK="${TASK:-gsm8k_llada_mini}"
TP="${TP:-2}"; GPUS="${GPUS:-0;1}"; PARALLEL="${PARALLEL:-tp}"
OUT="${OUT:-outputs/llada2_gsm8k_g${LENGTH}}"
PORT="${PORT:-23457}"
mkdir -p "$OUT"

echo "[llada2-baseline] model=$MODEL_PATH  gen=$LENGTH block=$BLOCK  thresholds: $THRESHOLDS  tp=$TP gpus=$GPUS  -> $OUT"

for thr in $THRESHOLDS; do
  outp="${OUT}/${TASK}_thr${thr}"
  echo; echo "==================== LLaDA-2.0  ${TASK}  thr=${thr}  gen=${LENGTH} ===================="
  python eval_dinfer_sglang.py --tasks "${TASK}" \
    --confirm_run_unsafe_code --model dInfer_eval \
    --model_args model_path="${MODEL_PATH}",gen_length="${LENGTH}",block_length="${BLOCK}",threshold="${thr}",low_threshold=0.0,show_speed=True,save_dir="${outp}",parallel_decoding=threshold,cache=prefix,warmup_times=0,use_compile=True,tp_size="${TP}",parallel="${PARALLEL}",cont_weight=0,use_credit=False,prefix_look=0,after_look=0,gpus="${GPUS}",model_type="${MODEL_TYPE}",use_bd=True,master_port="${PORT}",save_samples=True \
    --output_path "${outp}" --include_path tasks --apply_chat_template 2>&1 | tee "${OUT}/log_thr${thr}.txt"
done

echo; echo "[llada2-baseline] done. per-threshold results: ${OUT}/${TASK}_thr*/  logs: ${OUT}/log_thr*.txt"
echo "[llada2-baseline] grep accuracy:  grep -riE 'exact_match|acc' ${OUT}/${TASK}_thr*/  ;  speed in the logs."
