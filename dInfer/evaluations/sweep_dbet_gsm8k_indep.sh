#!/usr/bin/env bash
# Copyright 2026 University of Sydney. Apache-2.0.
#
# GSM8K sweep with INDEPENDENT heavy/draft thresholds + drafter top-k. Runs 16 configs:
#   4  heavy-only               : heavy_threshold in {0.5,0.7,0.8,0.9}
#   8  DBet (draft_top_k=1)      : heavy_threshold in {0.5,0.7,0.8,0.9} x draft_threshold in {0.8,0.9}
#   4  DBet (draft_top_k=2)      : heavy_threshold in {0.5,0.7,0.8,0.9}, draft_threshold=0.8
# gen_length 1024, 200 samples. Grades each with val_gsm8k.py and writes a summary table.
# Resumable (skips a config whose predictions already exist). Run from anywhere (cd's to dInfer root).
#   bash evaluations/sweep_dbet_gsm8k_indep.sh
# Override defaults via env vars, e.g.:
#   GEN=2048 LIMIT=500 DRAFTER=/abs/hf_ckpt HEAVY=/abs/DMax OUT=./sweep_out bash evaluations/sweep_dbet_gsm8k_indep.sh

set -u
cd "$(dirname "$0")/.."                          # -> dInfer/

DRAFTER="${DRAFTER:-../dFactory/dbet_outputs/checkpoints/global_step_20000/hf_ckpt/}"
HEAVY="${HEAVY:-../DMax-Math-16B-moe-merge}"
GEN="${GEN:-1024}"
LIMIT="${LIMIT:-200}"
BLOCK="${BLOCK:-32}"
OUT="${OUT:-./sweep_out_g1024}"
HEAVY_THRS="${HEAVY_THRS:-0.5 0.7 0.8 0.9}"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
printf "config\taccuracy\theavy/ex\tdraft/ex\twall/ex\ttok/s\n" > "$SUMMARY"

echo "[sweep] drafter=$DRAFTER  heavy=$HEAVY"
echo "[sweep] gen=$GEN limit=$LIMIT block=$BLOCK  heavy_thrs: $HEAVY_THRS  -> $OUT"

run_cfg () {                                      # $1=mode(heavy|dbet) $2=heavy_thr $3=draft_thr $4=draft_top_k
  local mode="$1" hthr="$2" dthr="$3" dk="$4" flag="" tag
  if [ "$mode" = heavy ]; then flag="--heavy_only"; tag="heavy_h${hthr}"
  else tag="dbet_h${hthr}_d${dthr}_k${dk}"; fi
  local preds="$OUT/preds_${tag}.jsonl" glog="$OUT/gen_${tag}.log" vlog="$OUT/grade_${tag}.log"
  echo; echo "==================== $tag ===================="

  if [ -s "$preds" ]; then
    echo "[sweep] $preds exists -> skip generation"
  else
    python evaluations/eval_dbet_gsm8k.py \
      --drafter_path "$DRAFTER" --heavy_path "$HEAVY" \
      --out_path "$preds" --limit "$LIMIT" --gen_length "$GEN" --block_length "$BLOCK" \
      --heavy_threshold "$hthr" --draft_threshold "$dthr" --draft_top_k "$dk" $flag 2>&1 | tee "$glog"
  fi

  python evaluations/val_gsm8k.py --pred-path "$preds" --limit "$LIMIT" 2>&1 | tee "$vlog"

  local acc heavy draft wall tps
  acc=$(grep -oE "Accuracy: [0-9.]+%" "$vlog" | tail -1 | grep -oE "[0-9.]+%")
  heavy=$(grep -oE "mean heavy/ex=[0-9.]+" "$glog" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  draft=$(grep -oE "draft/ex=[0-9.]+" "$glog" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  wall=$(grep -oE "wall/ex=[0-9.]+s" "$glog" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  tps=$(grep -oE "throughput=[0-9.]+ tok/s" "$glog" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "${acc:-NA}" "${heavy:-NA}" "${draft:-NA}" "${wall:-NA}" "${tps:-NA}" >> "$SUMMARY"
}

for h in $HEAVY_THRS; do run_cfg heavy "$h" 0 1; done                     # 4 heavy-only
for h in $HEAVY_THRS; do for d in 0.8 0.9; do run_cfg dbet "$h" "$d" 1; done; done   # 8 DBet k=1
for h in $HEAVY_THRS; do run_cfg dbet "$h" 0.8 2; done                    # 4 DBet k=2 (draft 0.8)

echo; echo "======================= SWEEP SUMMARY ======================="
column -t -s $'\t' "$SUMMARY"
echo "============================================================="
echo "[sweep] table: $SUMMARY   logs: $OUT/{gen,grade}_*.log   preds: $OUT/preds_*.jsonl"
