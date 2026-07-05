#!/usr/bin/env bash
# Copyright 2026 University of Sydney. Apache-2.0.
#
# GSM8K DBet sweep: 9 configs = heavy_threshold {0.9,0.7,0.5} x {draft 0.8, heavy-only, draft 0.9}, IN ORDER:
#   0.9xd0.8, 0.9 alone, 0.9xd0.9,  0.7xd0.8, 0.7 alone, 0.7xd0.9,  0.5xd0.8, 0.5 alone, 0.5xd0.9
# gen_length 512, 100 samples. After EACH config: grade (accuracy) + analyze (degeneration) + throughput report.
# Resumable (skips a config whose predictions already exist). Run from anywhere (cd's to dInfer root).
#   bash evaluations/sweep_dbet_gsm8k_indep.sh
# Override via env, e.g.:
#   DRAFTER=/abs/hf_ckpt HEAVY=/abs/DMax LIMIT=200 DK=2 DRAFT_EXTRA="--no_draft_fix" bash evaluations/sweep_dbet_gsm8k_indep.sh

set -u
cd "$(dirname "$0")/.."                          # -> dInfer/

DRAFTER="${DRAFTER:-../dFactory/dbet_outputs/checkpoints/global_step_20000/hf_ckpt/}"
HEAVY="${HEAVY:-../DMax-Math-16B-moe-merge}"
GEN="${GEN:-512}"
LIMIT="${LIMIT:-100}"
BLOCK="${BLOCK:-32}"
DK="${DK:-2}"                                    # draft_top_k for the DBet configs
DRAFT_EXTRA="${DRAFT_EXTRA:-}"                   # extra draft flags, e.g. "--no_draft_fix" / "--draft_committed_soft"
OUT="${OUT:-./sweep_out_g512}"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
printf "config\taccuracy\tdegen%%\tmean_tok\theavy/ex\tdraft/ex\twall/ex\ttok/s\n" > "$SUMMARY"

echo "[sweep] drafter=$DRAFTER  heavy=$HEAVY"
echo "[sweep] gen=$GEN limit=$LIMIT block=$BLOCK dk=$DK extra='$DRAFT_EXTRA'  -> $OUT"

run_cfg () {                                     # $1=mode(heavy|dbet)  $2=heavy_thr  $3=draft_thr
  local mode="$1" hthr="$2" dthr="$3" flag="" tag
  if [ "$mode" = heavy ]; then flag="--heavy_only"; tag="heavy_h${hthr}"
  else tag="dbet_h${hthr}_d${dthr}_k${DK}"; fi
  local preds="$OUT/preds_${tag}.jsonl" glog="$OUT/gen_${tag}.log" vlog="$OUT/grade_${tag}.log" alog="$OUT/degen_${tag}.log"
  echo; echo "==================== $tag ===================="

  # ---- generate ----
  if [ -s "$preds" ]; then
    echo "[sweep] $preds exists -> skip generation"
  else
    python evaluations/eval_dbet_gsm8k.py \
      --drafter_path "$DRAFTER" --heavy_path "$HEAVY" \
      --out_path "$preds" --limit "$LIMIT" --gen_length "$GEN" --block_length "$BLOCK" \
      --heavy_threshold "$hthr" --draft_threshold "$dthr" --draft_top_k "$DK" $flag $DRAFT_EXTRA 2>&1 | tee "$glog"
  fi

  # ---- accuracy ----
  echo "----- accuracy ($tag) -----"
  python evaluations/val_gsm8k.py --pred-path "$preds" --limit "$LIMIT" 2>&1 | tee "$vlog"
  # ---- degeneration ----
  echo "----- degeneration ($tag) -----"
  python evaluations/analyze_gen_lengths.py --dir "$OUT" --glob "preds_${tag}.jsonl" 2>&1 | tee "$alog"

  # ---- one-line report + summary row ----
  local acc degen mean heavy draft wall tps drow
  acc=$(grep -oE "Accuracy: [0-9.]+%" "$vlog" | tail -1 | grep -oE "[0-9.]+%")
  drow=$(grep "$tag" "$alog" | tail -1)
  degen=$(echo "$drow" | grep -oE "[0-9]+%" | sed -n '2p')       # 2nd % on the data row = degen% (1st = cap%)
  mean=$(echo "$drow"  | awk '{print $3}')                       # mean_tok
  heavy=$(grep -oE "mean heavy/ex=[0-9.]+" "$glog" | tail -1 | grep -oE "[0-9.]+")
  draft=$(grep -oE "draft/ex=[0-9.]+"     "$glog" | tail -1 | grep -oE "[0-9.]+")
  wall=$(grep -oE "wall/ex=[0-9.]+s"      "$glog" | tail -1 | grep -oE "[0-9.]+")
  tps=$(grep -oE "throughput=[0-9.]+ tok/s" "$glog" | tail -1 | grep -oE "[0-9.]+")
  echo ">>> [RESULT] $tag  acc=${acc:-NA}  degen=${degen:-NA}  |  heavy/ex=${heavy:-NA} draft/ex=${draft:-NA}  wall/ex=${wall:-NA}s  tok/s=${tps:-NA}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$tag" "${acc:-NA}" "${degen:-NA}" "${mean:-NA}" "${heavy:-NA}" "${draft:-NA}" "${wall:-NA}" "${tps:-NA}" >> "$SUMMARY"
}

# order: per heavy threshold -> draft 0.8, heavy-only, draft 0.9
for h in 0.9 0.7 0.5; do
  run_cfg dbet  "$h" 0.8
  run_cfg heavy "$h" 0
  run_cfg dbet  "$h" 0.9
done

echo; echo "======================= SWEEP SUMMARY ======================="
column -t -s $'\t' "$SUMMARY"
echo "============================================================="
echo "[sweep] table: $SUMMARY   logs: $OUT/{gen,grade,degen}_*.log   preds: $OUT/preds_*.jsonl"
