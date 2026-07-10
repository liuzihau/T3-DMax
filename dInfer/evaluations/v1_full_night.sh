#!/bin/bash
# Full-set (n=1319) confirmation of the v1-ckpt sweep winners (report_v1sweep_0710, n=200, +-2pt noise).
# Eager, faithful loop, heavy 0.5. 10 runs ~6h: dmax control + 5 candidates + 4 attribution rows
# (hard-mode isolates the mix's contribution; --no_draft_fix isolates FIX's, since dft0.5 winning
# contradicts the golden diag's net-negative verdict and needs a control before we believe it).
#
#   [DR=<v1 hf_ckpt>] bash evaluations/v1_full_night.sh

cd "$(dirname "$0")/.." || exit 1

HEAVY_EAGER=${HEAVY_EAGER:-../DMax-Math-16B-moe-merge}
DR=${DR:-../dFactory/dbet_outputs/checkpoints/global_step_30000-v1/hf_ckpt}
GEN=${GEN:-512}
H=${H:-0.5}
LIMIT=${LIMIT:-}
OUT=${OUT:-report_v1full_$(date +%m%d)}
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
      --gen_length "$GEN" --block_length 32 --heavy_threshold "$H")

# --- control ---
run "dmax_h${H}" "${BASE[@]}" --heavy_only

# --- the 5 candidates (best accuracy / best speed-at-iso from the n=200 sweep) ---
run "v1_dt0.7_dft0.5_k2_mix0.5"  "${BASE[@]}" --draft_threshold 0.7 --draft_fix_threshold 0.5 --draft_top_k 2 --draft_committed_mix 0.5
run "v1_dt0.7_dft0.9_k3_mixconf" "${BASE[@]}" --draft_threshold 0.7 --draft_fix_threshold 0.9 --draft_top_k 3 --draft_committed_mix conf
run "v1_dt0.7_dft0.9_k3_mix0.5"  "${BASE[@]}" --draft_threshold 0.7 --draft_fix_threshold 0.9 --draft_top_k 3 --draft_committed_mix 0.5
run "v1_dt0.7_dft0.7_k3_mix0.5"  "${BASE[@]}" --draft_threshold 0.7 --draft_fix_threshold 0.7 --draft_top_k 3 --draft_committed_mix 0.5
run "v1_dt0.9_dft0.9_k3_mixconf" "${BASE[@]}" --draft_threshold 0.9 --draft_fix_threshold 0.9 --draft_top_k 3 --draft_committed_mix conf

# --- attribution: mix contribution (same gates, HARD commits) ---
run "v1_dt0.7_dft0.5_k2_hard"    "${BASE[@]}" --draft_threshold 0.7 --draft_fix_threshold 0.5 --draft_top_k 2
run "v1_dt0.7_dft0.9_k3_hard"    "${BASE[@]}" --draft_threshold 0.7 --draft_fix_threshold 0.9 --draft_top_k 3

# --- attribution: FIX contribution (winners' gates, fix disabled) ---
run "v1_dt0.7_k2_mix0.5_nofix"   "${BASE[@]}" --draft_threshold 0.7 --draft_top_k 2 --draft_committed_mix 0.5 --no_draft_fix
run "v1_dt0.7_k3_mix0.5_nofix"   "${BASE[@]}" --draft_threshold 0.7 --draft_top_k 3 --draft_committed_mix 0.5 --no_draft_fix

python evaluations/summarize_baseline.py --dir "$OUT"
echo "=== [$(date +%H:%M:%S)] v1_full_night done -> $OUT/summary.tsv ==="
