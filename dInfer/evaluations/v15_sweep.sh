#!/bin/bash
# v1.5 (markov head) checkpoint sweep -- eager, faithful loop, heavy 0.5, draft_refine OFF by default.
# Reads the two decision numbers (experiment.md §v1full+v15train): tokens/call (dcommit/draft; walk working?
# target >=1.55 from 1.3) and heavy/ex vs the ~40 verification floor.
#
# Grid: extend th {0.5,0.6,0.7,0.8} x mix {conf,hard}, fix th 0.9, k2  = 8 rows @ LIMIT=200
# + dmax_h0.5 control + 2 ablations on the dt0.7/conf cell: --no_markov (walk vs +30k-steps attribution)
# and --draft_refine (does refine-round FIX shorten convergence, or delay Breakflag?).
#
#   [DR=<v15 hf_ckpt>] [LIMIT=200] [DTS=...] [MIXES=...] bash evaluations/v15_sweep.sh
# MIXES values: 'conf' (current-argmax DMax-style soft mix), a float (fixed blend), 'hard' (no mix flag).

cd "$(dirname "$0")/.." || exit 1

HEAVY_EAGER=${HEAVY_EAGER:-../DMax-Math-16B-moe-merge}
DR=${DR:-../dFactory/dbet_v15_outputs/checkpoints/global_step_30000/hf_ckpt}
LIMIT=${LIMIT:-200}
GEN=${GEN:-512}
H=${H:-0.5}
DTS=${DTS:-"0.5 0.6 0.7 0.8"}
DFT=${DFT:-0.9}
K=${K:-2}
MIXES=${MIXES:-"conf hard"}
OUT=${OUT:-report_v15sweep_$(date +%m%d)}
mkdir -p "$OUT"

run() {
  local tag=$1; shift
  if [ -s "$OUT/$tag.jsonl" ]; then echo "=== [skip] $tag ==="; return; fi
  echo "=== [$(date +%H:%M:%S)] $tag ==="
  if ! "$@" --out_path "$OUT/$tag.jsonl" --limit "$LIMIT" >"$OUT/$tag.log" 2>&1; then
    [ -e "$OUT/$tag.jsonl" ] && mv "$OUT/$tag.jsonl" "$OUT/$tag.jsonl.failed"
    echo "!!! $tag FAILED (see $OUT/$tag.log)"
  fi
}

BASE=(python evaluations/eval_dbet_gsm8k.py --heavy_path "$HEAVY_EAGER" --drafter_path "$DR" \
      --gen_length "$GEN" --block_length 32 --heavy_threshold "$H" \
      --draft_fix_threshold "$DFT" --draft_top_k "$K")

# --- control: pure DMax at the same heavy threshold (the Pareto yardstick, rerun in-session) ---
run "dmax_h${H}" python evaluations/eval_dbet_gsm8k.py --heavy_path "$HEAVY_EAGER" --drafter_path "$DR" \
    --gen_length "$GEN" --block_length 32 --heavy_threshold "$H" --heavy_only

# --- grid: aggressive-to-conservative extend gate x committed-input mode ---
for mix in $MIXES; do
  for dt in $DTS; do
    MIXARG=(); [ "$mix" != "hard" ] && MIXARG=(--draft_committed_mix "$mix")
    run "v15_dt${dt}_mix${mix}" "${BASE[@]}" --draft_threshold "$dt" "${MIXARG[@]}"
  done
done

# --- ablations on the reference cell (dt0.7 / conf) ---
run "v15_dt0.7_mixconf_nomarkov" "${BASE[@]}" --draft_threshold 0.7 --draft_committed_mix conf --no_markov
run "v15_dt0.7_mixconf_refine"   "${BASE[@]}" --draft_threshold 0.7 --draft_committed_mix conf --draft_refine

python evaluations/summarize_baseline.py --dir "$OUT"
echo "=== [$(date +%H:%M:%S)] v15_sweep done -> $OUT/summary.tsv ==="
