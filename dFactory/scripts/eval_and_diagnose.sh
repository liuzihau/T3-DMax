#!/usr/bin/env bash
# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# One-shot diagnostic + eval pipeline:
#   1. Inspect cross-attn weights and lm_head drift in the T3-D checkpoint
#   2. Full 500-sample CE eval of pure LLaDA-2.0-mini (baseline / teacher)
#   3. Full 500-sample CE eval of T3-D step-15000
#
# Run from the dFactory/ directory.
# Estimated total: ~50-60 min on a single H200 (diagnostic ~30s, each eval ~25 min).
# Outputs: three JSON files (one per step) + stdout logs.

set -euo pipefail

# ----------------------------------------------------------------------------
# Paths (edit if your run name or step are different)
# ----------------------------------------------------------------------------
RUN_NAME="${RUN_NAME:-t3d_1gpu_v6b_xattn_talk4_noiseramp_rolloutramp_lr1e4}"
STEP="${STEP:-15000}"

T3D_CKPT="./outputs/${RUN_NAME}/checkpoints/global_step_${STEP}/hf_ckpt"
LLADA_CKPT="../LLaDA2.0-mini-moe-merge"
VAL_PATH="./my_data/postprocess_train.jsonl"
SEED=42
VAL_TAIL=500

OUT_DIR="./eval_artifacts/${RUN_NAME}_step${STEP}"
mkdir -p "${OUT_DIR}"

# ----------------------------------------------------------------------------
# PYTHONPATH setup -- VeOmni + dFactory
# ----------------------------------------------------------------------------
export PYTHONPATH="$(pwd)/VeOmni:$(pwd):${PYTHONPATH:-}"

# Sanity check paths
if [[ ! -d "${T3D_CKPT}" ]]; then
    echo "ERROR: T3D_CKPT not found: ${T3D_CKPT}"
    echo "Verify your run name and step. Available runs:"
    find ./outputs -maxdepth 1 -type d 2>/dev/null | sort
    echo "Available step checkpoints under run:"
    find "./outputs/${RUN_NAME}" -name "global_step_*" -type d 2>/dev/null | sort
    exit 1
fi
if [[ ! -d "${LLADA_CKPT}" ]]; then
    echo "ERROR: LLADA_CKPT not found: ${LLADA_CKPT}"
    exit 1
fi
if [[ ! -f "${VAL_PATH}" ]]; then
    echo "ERROR: VAL_PATH not found: ${VAL_PATH}"
    exit 1
fi

echo "============================================================"
echo "  Eval + Diagnose Pipeline"
echo "============================================================"
echo "  T3D_CKPT:   ${T3D_CKPT}"
echo "  LLADA_CKPT: ${LLADA_CKPT}"
echo "  VAL_PATH:   ${VAL_PATH}  (seed=${SEED}, tail=${VAL_TAIL})"
echo "  OUT_DIR:    ${OUT_DIR}"
echo "============================================================"

# ----------------------------------------------------------------------------
# Step 1: Diagnostic (cross-attn weights + lm_head drift)
# ----------------------------------------------------------------------------
echo
echo "[1/3] Running weight diagnostic ..."
echo "------------------------------------------------------------"
python -u tasks/diagnose_t3d_weights.py \
    --t3d_ckpt "${T3D_CKPT}" \
    --llada_ckpt "${LLADA_CKPT}" \
    --talk_num_layers 4 \
    2>&1 | tee "${OUT_DIR}/diagnostic.log"

# ----------------------------------------------------------------------------
# Step 2: LLaDA full 500-sample CE baseline
# ----------------------------------------------------------------------------
echo
echo "[2/3] Running LLaDA full ${VAL_TAIL}-sample CE eval ..."
echo "------------------------------------------------------------"
python -u tasks/eval_ce_val.py \
    --model_path "${LLADA_CKPT}" \
    --model_type llada \
    --val_path "${VAL_PATH}" \
    --val_tail "${VAL_TAIL}" \
    --seed "${SEED}" \
    --output_json "${OUT_DIR}/eval_ce_llada.json" \
    2>&1 | tee "${OUT_DIR}/eval_llada.log"

# ----------------------------------------------------------------------------
# Step 3: T3-D full 500-sample CE
# ----------------------------------------------------------------------------
echo
echo "[3/3] Running T3-D full ${VAL_TAIL}-sample CE eval ..."
echo "------------------------------------------------------------"
python -u tasks/eval_ce_val.py \
    --model_path "${T3D_CKPT}" \
    --tokenizer_path "${LLADA_CKPT}" \
    --model_type t3d \
    --val_path "${VAL_PATH}" \
    --val_tail "${VAL_TAIL}" \
    --seed "${SEED}" \
    --output_json "${OUT_DIR}/eval_ce_t3d.json" \
    2>&1 | tee "${OUT_DIR}/eval_t3d.log"

# ----------------------------------------------------------------------------
# Done
# ----------------------------------------------------------------------------
echo
echo "============================================================"
echo "  Done. Artifacts in ${OUT_DIR}/"
echo "============================================================"
ls -la "${OUT_DIR}/"
