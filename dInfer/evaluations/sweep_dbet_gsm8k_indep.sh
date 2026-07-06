#!/usr/bin/env bash
# Copyright 2026 University of Sydney. Apache-2.0.
#
# GSM8K DBet FULL-REPORT battery. Explicit config list (mode heavy_thr draft_thr draft_top_k extra_flags),
# ordered by heavy-threshold GROUP (h0.9 first) so a partial run still yields a complete story. Per config it
# generates, grades (val_gsm8k -> accuracy), analyzes degeneration (analyze_gen_lengths --glob), reports
# throughput, prints a one-line RESULT, and appends a summary.tsv row. Resumable (skips existing preds).
#   bash evaluations/sweep_dbet_gsm8k_indep.sh
# Full GSM8K by default (LIMIT="" = all 1319). Override via env, e.g.:
#   LIMIT=200 DRAFTER=/abs/hf_ckpt HEAVY=/abs/DMax OUT=./report_out bash evaluations/sweep_dbet_gsm8k_indep.sh
# USE_CACHE=1 -> DBet configs use the incremental cross-block prefix-KV cache (--use_cache), tagged "_cache"
#   so cache and no-cache preds coexist; heavy baselines are cache-agnostic (shared). Run the cache A/B
#   (cache_ab_gsm8k.sh) FIRST and confirm iso-accuracy before launching a full USE_CACHE=1 sweep.

set -u
cd "$(dirname "$0")/.."                          # -> dInfer/
USE_CACHE="${USE_CACHE:-0}"

DRAFTER="${DRAFTER:-../dFactory/dbet_outputs/checkpoints/global_step_20000/hf_ckpt/}"
HEAVY="${HEAVY:-../DMax-Math-16B-moe-merge}"
GEN="${GEN:-512}"
LIMIT="${LIMIT:-}"                               # "" = full GSM8K test (1319)
BLOCK="${BLOCK:-32}"
OUT="${OUT:-./report_g512_full}"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
printf "config\taccuracy\tdegen%%\tmean_tok\theavy/ex\tdraft/ex\twall/ex\ttok/s\n" > "$SUMMARY"

LIM_FLAG=""; [ -n "$LIMIT" ] && LIM_FLAG="--limit $LIMIT"
echo "[sweep] drafter=$DRAFTER  heavy=$HEAVY"
echo "[sweep] gen=$GEN limit=${LIMIT:-ALL} block=$BLOCK use_cache=$USE_CACHE  -> $OUT"

# mode heavy_thr draft_thr draft_top_k extra_flags   (extra_flags may be empty)
CONFIGS=(
  # ---------- h0.9 group ----------
  "dbet  0.9 0.9 2"
  "heavy 0.9 0   1"
  "dbet  0.9 0.9 3"
  "dbet  0.9 0.8 3"
  # "dbet  0.9 0.8 2"
  # "dbet  0.9 0.9 2 --no_draft_fix"
  # "dbet  0.9 0.9 1"
  # "dbet  0.9 0.8 1"
  # ---------- h0.7 group ----------
  "dbet  0.7 0.9 2"
  "heavy 0.7 0   1"
  "dbet  0.7 0.9 3"
  "dbet  0.7 0.8 3"
  # "dbet  0.7 0.8 2"
  # "dbet  0.7 0.9 2 --no_draft_fix"
  # ---------- h0.5 group ----------
  "dbet  0.5 0.9 2"
  "heavy 0.5 0   1"
  "dbet  0.5 0.9 3"
  "dbet  0.5 0.8 3"
  # "dbet  0.5 0.8 2"
  # "dbet  0.5 0.9 2 --no_draft_fix"
)

run_cfg () {                                     # $1=mode $2=heavy_thr $3=draft_thr $4=draft_top_k $5..=extra
  local mode="$1" hthr="$2" dthr="$3" dk="$4"; shift 4; local extra="$*"
  local flag="" tag suffix="" csuffix=""
  [[ "$extra" == *--no_draft_fix* ]] && suffix="_nofix"
  if [ "$mode" = heavy ]; then flag="--heavy_only"; tag="heavy_h${hthr}"      # cache is DBet-only -> heavy shared
  else
    [ "$USE_CACHE" = 1 ] && { extra="$extra --use_cache"; csuffix="_cache"; }  # incremental prefix-KV cache
    tag="dbet_h${hthr}_d${dthr}_k${dk}${suffix}${csuffix}"
  fi
  local preds="$OUT/preds_${tag}.jsonl" glog="$OUT/gen_${tag}.log" vlog="$OUT/grade_${tag}.log" alog="$OUT/degen_${tag}.log"
  echo; echo "==================== $tag ===================="

  if [ -s "$preds" ]; then
    echo "[sweep] $preds exists -> skip generation"
  else
    python evaluations/eval_dbet_gsm8k.py \
      --drafter_path "$DRAFTER" --heavy_path "$HEAVY" \
      --out_path "$preds" $LIM_FLAG --gen_length "$GEN" --block_length "$BLOCK" \
      --heavy_threshold "$hthr" --draft_threshold "$dthr" --draft_top_k "$dk" $flag $extra 2>&1 | tee "$glog"
  fi

  echo "----- accuracy ($tag) -----"
  python evaluations/val_gsm8k.py --pred-path "$preds" $LIM_FLAG 2>&1 | tee "$vlog"
  echo "----- degeneration ($tag) -----"
  python evaluations/analyze_gen_lengths.py --dir "$OUT" --glob "preds_${tag}.jsonl" 2>&1 | tee "$alog"

  local acc degen mean heavy draft wall tps drow
  acc=$(grep -oE "Accuracy: [0-9.]+%" "$vlog" | tail -1 | grep -oE "[0-9.]+%")
  drow=$(grep "$tag" "$alog" | tail -1)
  degen=$(echo "$drow" | grep -oE "[0-9]+%" | sed -n '2p')       # 2nd % = degen% (1st = cap%)
  mean=$(echo "$drow"  | awk '{print $3}')
  # throughput/forwards straight from the PREDS jsonl (robust even when generation was skipped on resume)
  read heavy draft wall tps <<< "$(python3 -c "
import json
rows=[json.loads(l) for l in open('$preds')]
n=max(len(rows),1); tt=sum(r.get('wall_time',0.0) for r in rows) or 1e-9
print(f\"{sum(r.get('heavy_forwards',0) for r in rows)/n:.1f} {sum(r.get('draft_forwards',0) for r in rows)/n:.1f} {tt/n:.2f} {sum(r.get('gen_tokens',0) for r in rows)/tt:.1f}\")
" 2>/dev/null)"
  echo ">>> [RESULT] $tag  acc=${acc:-NA}  degen=${degen:-NA}  |  heavy/ex=${heavy:-NA} draft/ex=${draft:-NA}  wall/ex=${wall:-NA}s  tok/s=${tps:-NA}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$tag" "${acc:-NA}" "${degen:-NA}" "${mean:-NA}" "${heavy:-NA}" "${draft:-NA}" "${wall:-NA}" "${tps:-NA}" >> "$SUMMARY"
}

for cfg in "${CONFIGS[@]}"; do run_cfg $cfg; done

echo; echo "======================= SWEEP SUMMARY ======================="
column -t -s $'\t' "$SUMMARY"
echo "============================================================="
echo "[sweep] table: $SUMMARY   logs: $OUT/{gen,grade,degen}_*.log   preds: $OUT/preds_*.jsonl"
