#!/bin/bash
# FINAL full-set confirmation of the v1.5 Pareto candidates (report_v15sweep_0711, n=200):
#   dt0.6_mixhard 91.5%@1.28s and dt0.5_mixconf 91.0%@1.34s vs dmax_h0.5 90.5%@1.49s.
# 6 eager runs @ n=1319 (~3h): in-session dmax control, 4 candidate cells, 1 no-markov twin of the
# leader (walk attribution). Faithful loop, draft_refine off, fix 0.9, k2.
#
#   [DR=<v15 hf_ckpt>] bash evaluations/v15_final.sh

cd "$(dirname "$0")/.." || exit 1

HEAVY_EAGER=${HEAVY_EAGER:-../DMax-Math-16B-moe-merge}
DR=${DR:-../dFactory/dbet_v15_outputs/checkpoints/global_step_30000/hf_ckpt}
GEN=${GEN:-512}
H=${H:-0.5}
LIMIT=${LIMIT:-}
OUT=${OUT:-report_v15final_$(date +%m%d)}
mkdir -p "$OUT"

LIM=(); [ -n "$LIMIT" ] && LIM=(--limit "$LIMIT")

run() {
  local tag=$1; shift
  if [ -s "$OUT/$tag.jsonl" ]; then echo "=== [skip] $tag ==="; return; fi
  echo "=== [$(date +%H:%M:%S)] $tag ==="
  if ! "$@" --out_path "$OUT/$tag.jsonl" "${LIM[@]}" >"$OUT/$tag.log" 2>&1; then
    [ -e "$OUT/$tag.jsonl" ] && mv "$OUT/$tag.jsonl" "$OUT/$tag.jsonl.failed"
    echo "!!! $tag FAILED (see $OUT/$tag.log)"
  fi
}

BASE=(python evaluations/eval_dbet_gsm8k.py --heavy_path "$HEAVY_EAGER" --drafter_path "$DR" \
      --gen_length "$GEN" --block_length 32 --heavy_threshold "$H" \
      --draft_fix_threshold 0.9 --draft_top_k 2)

run "dmax_h${H}"            python evaluations/eval_dbet_gsm8k.py --heavy_path "$HEAVY_EAGER" \
    --drafter_path "$DR" --gen_length "$GEN" --block_length 32 --heavy_threshold "$H" --heavy_only

run "v15_dt0.6_mixhard"     "${BASE[@]}" --draft_threshold 0.6
run "v15_dt0.5_mixconf"     "${BASE[@]}" --draft_threshold 0.5 --draft_committed_mix conf
run "v15_dt0.6_mixconf"     "${BASE[@]}" --draft_threshold 0.6 --draft_committed_mix conf
run "v15_dt0.5_mixhard"     "${BASE[@]}" --draft_threshold 0.5   # the 87.5% outlier: real or n=200 noise?
run "v15_dt0.6_mixhard_nomarkov" "${BASE[@]}" --draft_threshold 0.6 --no_markov

python evaluations/summarize_baseline.py --dir "$OUT"
echo "=== [$(date +%H:%M:%S)] v15_final done -> $OUT/summary.tsv ==="
