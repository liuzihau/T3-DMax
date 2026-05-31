#!/usr/bin/env bash
# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# T3-D GSM8K evaluation — CANONICAL decode (matches diagnose_think_vs_talk.py v2:
# soft-embedding decode_uniform, grid-aligned blocks). Two stages: decode ->
# predictions jsonl, then grade with DMax's val_gsm8k.py (vendored here).
#
# Replaces the previous lm-eval-harness path (eval_dinfer_t3d.py), which drove
# the old hard-token / prompt-relative decode. Plain PyTorch, single GPU;
# accuracy is comparable to DMax, throughput is not (needs the vllm port).
#
# Usage:
#   bash eval_t3d_mini.sh                                   # full test set
#   LIMIT=20 bash eval_t3d_mini.sh                          # quick smoke
#   STEP=30000 RUN_NAME=t3d_xxx bash eval_t3d_mini.sh
#   MODEL_PATH=/abs/ckpt GEN_LENGTH=512 THRESHOLD=0.3 bash eval_t3d_mini.sh

set -euo pipefail

RUN_NAME="${RUN_NAME:-t3d_1gpu_v2_xattn_talk4_FROZEN_anchorDELTA_curriculum_sigma8010}"
STEP="${STEP:-15000}"

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"                  # .../dInfer/evaluations
T3DMAX_ROOT="$(cd "${EVAL_DIR}/../.." && pwd)"            # .../T3-DMax
DINFER_PYTHON="${T3DMAX_ROOT}/dInfer/python"
DFACTORY_ROOT="${T3DMAX_ROOT}/dFactory"

MODEL_PATH="${MODEL_PATH:-${DFACTORY_ROOT}/outputs/${RUN_NAME}/checkpoints/global_step_${STEP}/hf_ckpt}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${T3DMAX_ROOT}/LLaDA2.0-mini-moe-merge}"
OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_DIR}/outputs/gsm8k_t3d/${RUN_NAME}_step${STEP}}"

GEN_LENGTH="${GEN_LENGTH:-256}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
THRESHOLD="${THRESHOLD:-0.3}"
MAX_ITERS="${MAX_ITERS:-32}"
SOFT_TOP_K="${SOFT_TOP_K:-1}"
LIMIT="${LIMIT:-}"

mkdir -p "${OUTPUT_DIR}"
PRED_PATH="${OUTPUT_DIR}/predictions.jsonl"
export PYTHONPATH="${DINFER_PYTHON}:${DFACTORY_ROOT}:${DFACTORY_ROOT}/VeOmni:${PYTHONPATH:-}"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1

echo "[eval] model=${MODEL_PATH}"
echo "[eval] out=${PRED_PATH}  gen=${GEN_LENGTH} block=${BLOCK_LENGTH} threshold=${THRESHOLD}"

LIMIT_ARG=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

# Stage 1: decode
python "${EVAL_DIR}/eval_t3d_gsm8k.py" \
  --model_path "${MODEL_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --out_path "${PRED_PATH}" \
  --gen_length "${GEN_LENGTH}" \
  --block_length "${BLOCK_LENGTH}" \
  --threshold "${THRESHOLD}" \
  --max_iters_per_block "${MAX_ITERS}" \
  --soft_top_k "${SOFT_TOP_K}" \
  "${LIMIT_ARG[@]}"

# Stage 2: grade
echo "[eval] grading ${PRED_PATH}"
python "${EVAL_DIR}/val_gsm8k.py" --pred-path "${PRED_PATH}" "${LIMIT_ARG[@]}"
