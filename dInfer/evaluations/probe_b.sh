#!/usr/bin/env bash
# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# PREMISE PROBE — Variant B launcher (block-structured collect -> block-refiner fit).
# Tests whether a LIGHTWEIGHT block-level model + the static anchor + revealed
# neighbors can recover the converged tokens (the fair test of talk's mechanism).
#
# Re-runs the heavy decode (block-structured), so it costs another collection
# pass (~20 min on the 16B). Point MODEL_PATH at the same frozen full model used
# for the per-position probe.
#
# Usage:
#   MODEL_PATH=/abs/dmax_mini TAG=dmax_mini bash probe_b.sh

set -euo pipefail

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
T3DMAX_ROOT="$(cd "${EVAL_DIR}/../.." && pwd)"
DINFER_PYTHON="${T3DMAX_ROOT}/dInfer/python"
DFACTORY_ROOT="${T3DMAX_ROOT}/dFactory"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the frozen full model (DMax-finetuned checkpoint)}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
TAG="${TAG:-dmax_mini}"
OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_DIR}/outputs/premise_probe/${TAG}}"

NUM_PROMPTS="${NUM_PROMPTS:-200}"
GEN_LENGTH="${GEN_LENGTH:-256}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
THRESHOLD="${THRESHOLD:-0.3}"
MAX_BLOCKS="${MAX_BLOCKS:-8}"

mkdir -p "${OUTPUT_DIR}"
DATA_PATH="${OUTPUT_DIR}/probe_blocks.pt"
export PYTHONPATH="${DINFER_PYTHON}:${DFACTORY_ROOT}:${DFACTORY_ROOT}/VeOmni:${PYTHONPATH:-}"

echo "[probe-b] model=${MODEL_PATH}  collecting ${NUM_PROMPTS} prompts (block-structured) -> ${DATA_PATH}"
python "${EVAL_DIR}/probe_collect_blocks.py" \
  --model_path "${MODEL_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --out_path "${DATA_PATH}" \
  --num_prompts "${NUM_PROMPTS}" \
  --gen_length "${GEN_LENGTH}" \
  --block_length "${BLOCK_LENGTH}" \
  --threshold "${THRESHOLD}" \
  --max_blocks "${MAX_BLOCKS}"

echo "[probe-b] fitting block-level refiner ..."
python "${EVAL_DIR}/probe_fit_b.py" --data "${DATA_PATH}"
