#!/bin/bash
# Nightly GSM8K baseline rebuild (post-fixes, 2026-07-09): LLaDA-2.0 (fixed steps 16/9/6) vs DMax
# (h 0.5/0.7/0.9) vs DBet (h 0.5/0.7/0.9, d0.9 dfix0.9 k2), each in BOTH stacks (sglang cached+graph,
# eager). 18 runs. SGLang first (fast, numbers early), eager LLaDA last (slowest). A failed run logs
# and the sweep continues.
#
#   DR=/path/to/drafter [HEAVY=../DMax-Math-16B] [LIMIT=500] [GEN=512] bash evaluations/baseline_night.sh
#
# Empty LIMIT = full GSM8K test (1319). Everything lands in $OUT (jsonl + .log per run), then
# summarize_baseline.py writes summary.tsv (accuracy + heavy/ex + draft/ex + wall/ex + tok/s).

cd "$(dirname "$0")/.." || exit 1                       # dInfer root (evaluations/ paths below)

HEAVY=${HEAVY:-../DMax-Math-16B}
DR=${DR:?set DR to the drafter ckpt path}
LIMIT=${LIMIT:-}
GEN=${GEN:-512}
OUT=${OUT:-report_baseline_$(date +%m%d)}
mkdir -p "$OUT"

LIM=()
[ -n "$LIMIT" ] && LIM=(--limit "$LIMIT")

run() {  # run <tag> <cmd...>  — a failed run's partial jsonl is moved aside so a rerun retries it
  local tag=$1; shift
  if [ -s "$OUT/$tag.jsonl" ]; then echo "=== [skip] $tag (jsonl exists) ==="; return; fi
  echo "=== [$(date +%H:%M:%S)] $tag ==="
  if ! "$@" --out_path "$OUT/$tag.jsonl" "${LIM[@]}" >"$OUT/$tag.log" 2>&1; then
    [ -e "$OUT/$tag.jsonl" ] && mv "$OUT/$tag.jsonl" "$OUT/$tag.jsonl.failed"
    echo "!!! $tag FAILED (see $OUT/$tag.log)"
  fi
}

SGL=(python evaluations/eval_dbet_sglang_cached_gsm8k.py --heavy_path "$HEAVY" \
     --gen_length "$GEN" --block_length 32 --cuda_graph)
EAGER=(python evaluations/eval_dbet_gsm8k.py --heavy_path "$HEAVY" --drafter_path "$DR" \
       --gen_length "$GEN" --block_length 32)

# ---------- SGLANG (fast) ----------
for s in 16 9 6; do
  run "sglang_llada_s${s}" "${SGL[@]}" --no_draft --decoder fixed --steps "$s"
done
for h in 0.5 0.7 0.9; do
  run "sglang_dmax_h${h}" "${SGL[@]}" --no_draft --heavy_threshold "$h"
done
for h in 0.5 0.7 0.9; do
  run "sglang_dbet_h${h}" "${SGL[@]}" --drafter_path "$DR" --compile_draft \
      --heavy_threshold "$h" --draft_threshold 0.9 --draft_fix_threshold 0.9 --draft_top_k 2
done

# ---------- EAGER ----------
for h in 0.5 0.7 0.9; do
  run "eager_dmax_h${h}" "${EAGER[@]}" --heavy_only --heavy_threshold "$h"
done
for h in 0.5 0.7 0.9; do
  # eager decode has ONE draft threshold gating both EXTEND and FIX -> 0.9 == d0.9+dfix0.9
  run "eager_dbet_h${h}" "${EAGER[@]}" --heavy_threshold "$h" --draft_threshold 0.9 --draft_top_k 2
done
for s in 16 9 6; do                                     # slowest family last
  run "eager_llada_s${s}" python evaluations/eval_llada_gsm8k.py --heavy_path "$HEAVY" \
      --gen_length "$GEN" --block_length 32 --steps "$s"
done

# ---------- SUMMARY ----------
python evaluations/summarize_baseline.py --dir "$OUT"
echo "=== [$(date +%H:%M:%S)] baseline_night done -> $OUT/summary.tsv ==="
