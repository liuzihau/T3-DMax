#!/usr/bin/env bash
# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# One-shot GSM8K eval: generate predictions with T3-D (and optionally LLaDA baseline),
# then grade both with DMax's val_gsm8k.py (vendored as tasks/gsm8k_grade.py).
#
# Defaults to step-15000 of v6e. Override via env vars.
#
# Usage:
#   bash scripts/gsm8k_eval.sh                                  # uses defaults
#   STEP=30000 bash scripts/gsm8k_eval.sh                       # different checkpoint
#   LIMIT=20 bash scripts/gsm8k_eval.sh                         # quick smoke test
#   SKIP_LLADA=1 bash scripts/gsm8k_eval.sh                     # T3-D only
#   SKIP_T3D=1 bash scripts/gsm8k_eval.sh                       # LLaDA only

set -euo pipefail

# ----------------------------------------------------------------------------
# Paths (override via env)
# ----------------------------------------------------------------------------
RUN_NAME="${RUN_NAME:-t3d_1gpu_v6e_xattn_talk4_FROZEN_anchorDELTA_multiITER5_lr1e4}"
STEP="${STEP:-15000}"

T3D_CKPT="./outputs/${RUN_NAME}/checkpoints/global_step_${STEP}/hf_ckpt"
LLADA_CKPT="../LLaDA2.0-mini-moe-merge"

OUT_DIR="./eval_artifacts/gsm8k/${RUN_NAME}_step${STEP}"
mkdir -p "${OUT_DIR}"

# Generation knobs
GEN_LENGTH="${GEN_LENGTH:-512}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
MAX_ITER_PER_BLOCK="${MAX_ITER_PER_BLOCK:-8}"
THRESHOLD="${THRESHOLD:-0.5}"
LIMIT="${LIMIT:-}"   # empty = full test set (~1.3k); set to e.g. 20 for smoke

LIMIT_ARG=""
if [[ -n "${LIMIT}" ]]; then
    LIMIT_ARG="--limit ${LIMIT}"
fi

# ----------------------------------------------------------------------------
# PYTHONPATH
# ----------------------------------------------------------------------------
export PYTHONPATH="$(pwd)/VeOmni:$(pwd):${PYTHONPATH:-}"

echo "============================================================"
echo "  GSM8K Eval Pipeline"
echo "============================================================"
echo "  RUN_NAME:           ${RUN_NAME}"
echo "  STEP:               ${STEP}"
echo "  T3D_CKPT:           ${T3D_CKPT}"
echo "  LLADA_CKPT:         ${LLADA_CKPT}"
echo "  OUT_DIR:            ${OUT_DIR}"
echo "  GEN_LENGTH:         ${GEN_LENGTH}"
echo "  BLOCK_SIZE:         ${BLOCK_SIZE}"
echo "  MAX_ITER_PER_BLOCK: ${MAX_ITER_PER_BLOCK}"
echo "  THRESHOLD:          ${THRESHOLD}"
if [[ -n "${LIMIT}" ]]; then
    echo "  LIMIT:              ${LIMIT} (smoke mode)"
fi
echo "============================================================"

# ----------------------------------------------------------------------------
# 1. LLaDA baseline generate + grade
# ----------------------------------------------------------------------------
if [[ -z "${SKIP_LLADA:-}" ]]; then
    if [[ ! -d "${LLADA_CKPT}" ]]; then
        echo "ERROR: LLaDA checkpoint not found at ${LLADA_CKPT}"
        exit 1
    fi
    echo
    echo "[1/2] LLaDA baseline ..."
    echo "------------------------------------------------------------"
    LLADA_PRED="${OUT_DIR}/predictions_llada.jsonl"
    python -u tasks/gsm8k_generate.py \
        --model_path "${LLADA_CKPT}" \
        --model_type llada \
        --output_path "${LLADA_PRED}" \
        --gen_length "${GEN_LENGTH}" \
        --block_size "${BLOCK_SIZE}" \
        --max_iter_per_block "${MAX_ITER_PER_BLOCK}" \
        --threshold "${THRESHOLD}" \
        ${LIMIT_ARG} \
        2>&1 | tee "${OUT_DIR}/generate_llada.log"

    echo
    echo "[grading LLaDA] ..."
    python -u tasks/gsm8k_grade.py \
        --pred-path "${LLADA_PRED}" \
        2>&1 | tee "${OUT_DIR}/grade_llada.log"
fi

# ----------------------------------------------------------------------------
# 2. T3-D generate + grade
# ----------------------------------------------------------------------------
if [[ -z "${SKIP_T3D:-}" ]]; then
    if [[ ! -d "${T3D_CKPT}" ]]; then
        echo "ERROR: T3-D checkpoint not found at ${T3D_CKPT}"
        echo "Available run/step combos under outputs/:"
        find ./outputs -name "hf_ckpt" -type d 2>/dev/null | sort
        exit 1
    fi
    echo
    echo "[2/2] T3-D ..."
    echo "------------------------------------------------------------"
    T3D_PRED="${OUT_DIR}/predictions_t3d.jsonl"
    python -u tasks/gsm8k_generate.py \
        --model_path "${T3D_CKPT}" \
        --tokenizer_path "${LLADA_CKPT}" \
        --model_type t3d \
        --output_path "${T3D_PRED}" \
        --gen_length "${GEN_LENGTH}" \
        --block_size "${BLOCK_SIZE}" \
        --max_iter_per_block "${MAX_ITER_PER_BLOCK}" \
        --threshold "${THRESHOLD}" \
        ${LIMIT_ARG} \
        2>&1 | tee "${OUT_DIR}/generate_t3d.log"

    echo
    echo "[grading T3-D] ..."
    python -u tasks/gsm8k_grade.py \
        --pred-path "${T3D_PRED}" \
        2>&1 | tee "${OUT_DIR}/grade_t3d.log"
fi

echo
echo "============================================================"
echo "  Done. Artifacts in ${OUT_DIR}/"
echo "============================================================"
ls -la "${OUT_DIR}/"
