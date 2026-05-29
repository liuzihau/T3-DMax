#!/usr/bin/env bash
# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# T3-D GSM8K evaluation launcher. Mirrors DMax's eval_llada_mini.sh but
# dispatches to the t3d_eval lm-eval model defined in eval_dinfer_t3d.py.
#
# Defaults to GSM8K on v6e step-15000. Override via env vars:
#   STEP=30000 RUN_NAME=t3d_xxx bash eval_t3d_mini.sh
#   THRESHOLD=0.95 bash eval_t3d_mini.sh
#   LIMIT=20 bash eval_t3d_mini.sh        # quick smoke test (lm-eval --limit)

set -euo pipefail

# ----------------------------------------------------------------------------
# Paths (override via env)
# ----------------------------------------------------------------------------
RUN_NAME="${RUN_NAME:-t3d_1gpu_v6e_xattn_talk4_FROZEN_anchorDELTA_multiITER5_lr1e4}"
STEP="${STEP:-15000}"

T3DMAX_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"        # .../T3-DMax
DINFER_ROOT="${T3DMAX_ROOT}/dInfer"
DFACTORY_ROOT="${T3DMAX_ROOT}/dFactory"

MODEL_PATH="${MODEL_PATH:-${DFACTORY_ROOT}/outputs/${RUN_NAME}/checkpoints/global_step_${STEP}/hf_ckpt}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${T3DMAX_ROOT}/../LLaDA2.0-mini-moe-merge}"
OUTPUT_DIR="${OUTPUT_DIR:-${DFACTORY_ROOT}/eval_artifacts/gsm8k_lm_eval/${RUN_NAME}_step${STEP}}"

# ----------------------------------------------------------------------------
# Decoding knobs (defaults track DMax's gsm8k-llada-mini.yaml + paper)
# ----------------------------------------------------------------------------
GEN_LENGTH="${GEN_LENGTH:-512}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
THRESHOLD="${THRESHOLD:-0.9}"
MAX_ITER="${MAX_ITER:-32}"
TASKS="${TASKS:-gsm8k_llada_mini}"

# Optional: pass through to lm-eval-harness --limit.
LIMIT="${LIMIT:-}"

mkdir -p "${OUTPUT_DIR}"

# ----------------------------------------------------------------------------
# PYTHONPATH wiring
# ----------------------------------------------------------------------------
# - dInfer/python : where `dinfer.*` modules live
# - dFactory      : where the T3-D training model + masks live (the shim adds
#                   this automatically, but exporting it makes failures obvious)
# - dFactory/VeOmni: VeOmni framework (used by the model registry path-rewrite)
export PYTHONPATH="${DINFER_ROOT}/python:${DFACTORY_ROOT}:${DFACTORY_ROOT}/VeOmni:${PYTHONPATH:-}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1

# ----------------------------------------------------------------------------
# Launch
# ----------------------------------------------------------------------------
cd "${DINFER_ROOT}/evaluations"

LIMIT_FLAG=""
if [[ -n "${LIMIT}" ]]; then
    LIMIT_FLAG="--limit ${LIMIT}"
fi

python eval_dinfer_t3d.py \
    --tasks "${TASKS}" \
    --confirm_run_unsafe_code \
    --model t3d_eval \
    --model_args "model_path=${MODEL_PATH},tokenizer_path=${TOKENIZER_PATH},gen_length=${GEN_LENGTH},block_length=${BLOCK_LENGTH},threshold=${THRESHOLD},max_iter_per_block=${MAX_ITER},save_dir=${OUTPUT_DIR},save_samples=True,show_speed=True" \
    --output_path "${OUTPUT_DIR}/lm_eval_results.json" \
    --include_path "${DINFER_ROOT}/evaluations/tasks" \
    --apply_chat_template \
    ${LIMIT_FLAG}

echo ""
echo "[T3-D eval] done. Results: ${OUTPUT_DIR}/lm_eval_results.json"
echo "[T3-D eval] per-sample log:  ${OUTPUT_DIR}/rank_0.jsonl"
