#!/bin/bash
# --no_force_first A/B on the SURVIVING full-set cells (v15_final n=1319 verdict: the n=200 Pareto rows
# regressed; markov's walk costs ~0.2s/ex at eager batch=1 and its positionwise gains don't convert ->
# the live candidates are NOMARKOV dt0.5/0.6, holding iso-accuracy-within-noise at ~15% less wall).
# The profiler showed 14% of chains open with a FORCED below-gate token accepting at 0.36 -- dropping the
# force removes pure repair load; this run measures whether that converts to accuracy/wall.
#
# 7 eager runs @ n=1319 (~3.5h): in-session dmax control + 3 cells x {force (default), noforce}:
#   dt0.6_mixconf_k2_nomarkov   (best surviving accuracy: 90.60 @ 1.39)
#   dt0.5_mixhard_k2_nomarkov   (best surviving speed:    90.45 @ 1.35)
#   dt0.5_mixconf_k2 (markov)   (best markov row 90.45 -- does no-force interact with the walk's k0 bias?)
#
#   [DR=<v15 hf_ckpt>] [LIMIT=] bash evaluations/noforce_test.sh

cd "$(dirname "$0")/.." || exit 1

HEAVY_EAGER=${HEAVY_EAGER:-../DMax-Math-16B-moe-merge}
DR=${DR:-../dFactory/dbet_v15_outputs/checkpoints/global_step_30000/hf_ckpt}
GEN=${GEN:-512}
H=${H:-0.5}
LIMIT=${LIMIT:-}
OUT=${OUT:-report_noforce_$(date +%m%d)}
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

# in-session control (wall varies ~15% between sessions -- never compare across sheets)
run "dmax_h${H}" python evaluations/eval_dbet_gsm8k.py --heavy_path "$HEAVY_EAGER" --drafter_path "$DR" \
    --gen_length "$GEN" --block_length 32 --heavy_threshold "$H" --heavy_only

# cell 1: best surviving accuracy (nomarkov)
run "dt0.6_mixconf_k2_nm_force"   "${BASE[@]}" --draft_threshold 0.6 --draft_committed_mix conf --no_markov
run "dt0.6_mixconf_k2_nm_noforce" "${BASE[@]}" --draft_threshold 0.6 --draft_committed_mix conf --no_markov --no_force_first

# cell 2: best surviving speed (nomarkov)
run "dt0.5_mixhard_k2_nm_force"   "${BASE[@]}" --draft_threshold 0.5 --no_markov
run "dt0.5_mixhard_k2_nm_noforce" "${BASE[@]}" --draft_threshold 0.5 --no_markov --no_force_first

# cell 3: best markov row (walk x no-force interaction)
run "dt0.5_mixconf_k2_mk_force"   "${BASE[@]}" --draft_threshold 0.5 --draft_committed_mix conf
run "dt0.5_mixconf_k2_mk_noforce" "${BASE[@]}" --draft_threshold 0.5 --draft_committed_mix conf --no_force_first

python evaluations/summarize_baseline.py --dir "$OUT"
echo "=== [$(date +%H:%M:%S)] noforce_test done -> $OUT/summary.tsv ==="
