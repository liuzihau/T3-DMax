#!/usr/bin/env bash
# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# PREMISE PROBE launcher (collect -> fit). Tests whether a heavy dLLM's iter-0
# representation already encodes its converged answer (the T3-D hypothesis).
#
# Point MODEL_PATH at the DMax-finetuned checkpoint (the proposed backbone). Run
# it ALSO on LLaDA2.0-mini-moe-merge (TAG=llada_mini) to compare whether the
# DMax fine-tune makes the first representation richer.
#
# Usage:
#   MODEL_PATH=/abs/dmax_mini TAG=dmax_mini bash probe.sh
#   MODEL_PATH=/abs/dmax_mini NUM_PROMPTS=400 bash probe.sh

set -euo pipefail

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"                  # .../dInfer/evaluations
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
DATA_PATH="${OUTPUT_DIR}/probe_data.pt"
export PYTHONPATH="${DINFER_PYTHON}:${DFACTORY_ROOT}:${DFACTORY_ROOT}/VeOmni:${PYTHONPATH:-}"

echo "[probe] model=${MODEL_PATH}  collecting ${NUM_PROMPTS} prompts -> ${DATA_PATH}"
python "${EVAL_DIR}/probe_collect_t3d.py" \
  --model_path "${MODEL_PATH}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --out_path "${DATA_PATH}" \
  --num_prompts "${NUM_PROMPTS}" \
  --gen_length "${GEN_LENGTH}" \
  --block_length "${BLOCK_LENGTH}" \
  --threshold "${THRESHOLD}" \
  --max_blocks "${MAX_BLOCKS}"

echo "[probe] fitting lightweight heads ..."
python "${EVAL_DIR}/probe_fit.py" --data "${DATA_PATH}"
