#!/usr/bin/env bash
# Copyright 2026 University of Sydney. Apache-2.0.
#
# GSM8K threshold sweep: for each threshold, run BOTH heavy-only and DBet, then grade.
# The threshold is applied to BOTH the heavy commit (--heavy_threshold) and the drafter
# gate (--draft_threshold). Resumable (skips a config whose predictions already exist)
# and prints a summary table of accuracy + speed at the end.
#
# Run from anywhere; it cd's to the dInfer root. Long (~8 gen runs x 500 ex); use tmux/nohup.
#   bash evaluations/sweep_dbet_gsm8k.sh
# Override defaults via env vars, e.g.:
#   LIMIT=500 THRESHOLDS="0.5 0.7 0.8 0.9" DRAFTER=... HEAVY=... bash evaluations/sweep_dbet_gsm8k.sh

set -u
cd "$(dirname "$0")/.."                          # -> dInfer/

DRAFTER="${DRAFTER:-../dFactory/dbet_outputs/checkpoints/global_step_20000/hf_ckpt/}"
HEAVY="${HEAVY:-../DMax-Math-16B-moe-merge}"
LIMIT="${LIMIT:-500}"
GEN="${GEN:-256}"
BLOCK="${BLOCK:-32}"
THRESHOLDS="${THRESHOLDS:-0.5 0.7 0.8 0.9}"
OUT="${OUT:-./sweep_out}"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
printf "config\taccuracy\theavy/ex\tdraft/ex\twall/ex\ttok/s\n" > "$SUMMARY"

echo "[sweep] drafter=$DRAFTER"
echo "[sweep] heavy=$HEAVY  limit=$LIMIT gen=$GEN block=$BLOCK  thresholds: $THRESHOLDS"
echo "[sweep] outputs -> $OUT"

run_one () {                                      # $1=mode(heavy|dbet) $2=threshold $3=extra flag
  local mode="$1" thr="$2" flag="$3"
  local tag="${mode}_thr${thr}"
  local preds="$OUT/preds_${tag}.jsonl"
  local glog="$OUT/gen_${tag}.log"
  local vlog="$OUT/grade_${tag}.log"
  echo; echo "==================== $tag ===================="

  if [[ -s "$preds" ]]; then
    echo "[sweep] $preds exists -> skip generation"
  else
    python evaluations/eval_dbet_gsm8k.py \
      --drafter_path "$DRAFTER" --heavy_path "$HEAVY" \
      --out_path "$preds" --limit "$LIMIT" --gen_length "$GEN" --block_length "$BLOCK" \
      --heavy_threshold "$thr" --draft_threshold "$thr" $flag 2>&1 | tee "$glog"
  fi

  python evaluations/val_gsm8k.py --pred-path "$preds" --limit "$LIMIT" 2>&1 | tee "$vlog"

  # scrape key metrics into the summary (from the persisted logs; robust to skipped gen)
  local acc heavy draft wall tps
  acc=$(grep -oE "Accuracy: [0-9.]+%" "$vlog" | tail -1 | grep -oE "[0-9.]+%")
  heavy=$(grep -oE "mean heavy/ex=[0-9.]+" "$glog" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  draft=$(grep -oE "draft/ex=[0-9.]+" "$glog" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  wall=$(grep -oE "wall/ex=[0-9.]+s" "$glog" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  tps=$(grep -oE "throughput=[0-9.]+ tok/s" "$glog" 2>/dev/null | tail -1 | grep -oE "[0-9.]+")
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "${acc:-NA}" "${heavy:-NA}" "${draft:-NA}" "${wall:-NA}" "${tps:-NA}" >> "$SUMMARY"
}

for thr in $THRESHOLDS; do
  run_one heavy "$thr" --heavy_only
  run_one dbet  "$thr" ""
done

echo; echo "======================= SWEEP SUMMARY ======================="
column -t -s $'\t' "$SUMMARY"
echo "============================================================="
echo "[sweep] full table: $SUMMARY   logs: $OUT/{gen,grade}_*.log   preds: $OUT/preds_*.jsonl"
