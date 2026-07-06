#!/usr/bin/env bash
# Copyright 2026 University of Sydney. Apache-2.0.
#
# Unattended proof that the DBet prefix-KV cache is CORRECT and the earlier divergence was float
# non-associativity (bf16). Runs, logs, and summarizes:
#   forward-level (fast, single forward -> shows the magnitude of each source):
#     (a) bf16 raw            -> fused-MoE grouping + flash-attn tiling   (max|Δ| ~ 0.1)
#     (b) bf16 --exact_moe    -> flash-attn tiling only                   (smaller)
#     (c) fp32 --exact_moe    -> near-zero                                (~1e-4  => it's precision)
#   decode-level (slow, full generation -> the verdict):
#     (1) bf16 --exact_moe --math_attn -> both non-assoc sources removed  => expect 10/10 BIT-IDENTICAL
#     (2) fp32 --exact_moe             -> tokens should match             (~64GB; may OOM -> skipped)
# Run:  DRAFTER=/abs/hf_ckpt HEAVY=/abs/DMax bash evaluations/prove_cache.sh
set -u
cd "$(dirname "$0")/.."                                   # -> dInfer/
export PYTHONPATH="$(pwd)/python:${PYTHONPATH:-}"

DRAFTER="${DRAFTER:-../dFactory/dbet_outputs/checkpoints/global_step_30000-v1/hf_ckpt}"
HEAVY="${HEAVY:-../DMax-Math-16B-moe-merge}"
LIMIT="${LIMIT:-10}"
GEN="${GEN:-512}"
OUT="${OUT:-cache_proof}"
mkdir -p "$OUT"
DIAG="--drafter_path $DRAFTER --heavy_path $HEAVY --block_length 32"
VAL="--drafter_path $DRAFTER --heavy_path $HEAVY --limit $LIMIT --gen_length $GEN --heavy_threshold 0.9 --draft_threshold 0.9 --draft_top_k 2"
run () { echo; echo "==================== $1 ===================="; shift; ( "$@" ) 2>&1 | tee "$LOG"; }

echo "[prove_cache] drafter=$DRAFTER heavy=$HEAVY limit=$LIMIT gen=$GEN -> $OUT"

LOG="$OUT/a_diag_bf16_raw.log"
run "(a) forward-level  bf16 RAW  (MoE + attn non-assoc)"        python evaluations/diag_cache.py $DIAG --dtype bfloat16
LOG="$OUT/b_diag_bf16_exactmoe.log"
run "(b) forward-level  bf16 --exact_moe  (attn-tiling only)"    python evaluations/diag_cache.py $DIAG --dtype bfloat16 --exact_moe
LOG="$OUT/c_diag_fp32_exactmoe.log"
run "(c) forward-level  fp32 --exact_moe  (near-zero => precision)" python evaluations/diag_cache.py $DIAG --dtype float32 --exact_moe

LOG="$OUT/1_validate_exactmoe_mathattn.log"
run "(1) DECODE  bf16 --exact_moe --math_attn  (expect 10/10 BIT-IDENTICAL)"  python evaluations/validate_cache.py $VAL --exact_moe --math_attn
LOG="$OUT/2_validate_fp32_exactmoe.log"
run "(2) DECODE  fp32 --exact_moe  (tokens should match; may OOM)"            python evaluations/validate_cache.py $VAL --exact_moe --dtype float32

echo; echo "======================= SUMMARY ======================="
for f in a b c; do
  L=$(ls "$OUT/${f}_"*.log 2>/dev/null); [ -n "$L" ] && printf "%s : %s\n" "$f" "$(grep -m1 'block logits' "$L" 2>/dev/null || echo '(no output / failed)')"
done
for f in 1 2; do
  L=$(ls "$OUT/${f}_"*.log 2>/dev/null); [ -n "$L" ] && printf "%s : %s\n" "$f" "$(grep -Eom1 '[0-9]+/[0-9]+ identical.*' "$L" 2>/dev/null || echo '(no verdict / OOM / failed)')"
done
echo "======================================================="
echo "[prove_cache] logs in $OUT/  |  KEY: (1) 10/10 => cache correct, divergence was pure bf16 non-associativity"
