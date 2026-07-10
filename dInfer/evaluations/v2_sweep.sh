#!/bin/bash
# TRAIN-V2 checkpoint sweep (eager, faithful loop): fixed heavy 0.5; grid over the drafter knobs.
#   draft_th x draft_fix_th x top_k x committed-mix  =  {0.7,0.8,0.9} x {0.5,0.7,0.9} x {2,3} x {conf,0.5}
# 36 runs @ LIMIT=200 (~4-5 min each eager) + a DMax h0.5 control row. Same summary sheet as
# baseline_night (summarize_baseline.py). Resumable: existing jsonls are skipped, failures moved aside.
#
#   [DR=<v2 hf_ckpt>] [LIMIT=200] [MIXES="conf 0.5"] bash evaluations/v2_sweep.sh
#
# MIXES values: 'conf' (DMax confidence-weighted soft embed), a float (fixed H/M interpolation),
# or 'hard' (route H: plain hard commits, no mix flag).

cd "$(dirname "$0")/.." || exit 1

HEAVY_EAGER=${HEAVY_EAGER:-../DMax-Math-16B-moe-merge}
DR=${DR:-../dFactory/dbet_v2_outputs/checkpoints/global_step_5000/hf_ckpt}
LIMIT=${LIMIT:-200}
GEN=${GEN:-512}
H=${H:-0.5}
DTS=${DTS:-"0.7 0.8 0.9"}
DFTS=${DFTS:-"0.5 0.7 0.9"}
KS=${KS:-"2 3"}
MIXES=${MIXES:-"conf 0.5"}
OUT=${OUT:-report_v2sweep_$(date +%m%d)}
mkdir -p "$OUT"

run() {  # run <tag> <cmd...>
  local tag=$1; shift
  if [ -s "$OUT/$tag.jsonl" ]; then echo "=== [skip] $tag ==="; return; fi
  echo "=== [$(date +%H:%M:%S)] $tag ==="
  if ! "$@" --out_path "$OUT/$tag.jsonl" --limit "$LIMIT" >"$OUT/$tag.log" 2>&1; then
    [ -e "$OUT/$tag.jsonl" ] && mv "$OUT/$tag.jsonl" "$OUT/$tag.jsonl.failed"
    echo "!!! $tag FAILED (see $OUT/$tag.log)"
  fi
}

BASE=(python evaluations/eval_dbet_gsm8k.py --heavy_path "$HEAVY_EAGER" --drafter_path "$DR" \
      --gen_length "$GEN" --block_length 32 --heavy_threshold "$H")

# control: pure DMax at the same heavy threshold (faithful loop), the Pareto yardstick
run "dmax_h${H}" "${BASE[@]}" --heavy_only

for mix in $MIXES; do
  for dt in $DTS; do
    for dft in $DFTS; do
      for k in $KS; do
        tag="v2_dt${dt}_dft${dft}_k${k}_mix${mix}"
        MIXARG=(); [ "$mix" != "hard" ] && MIXARG=(--draft_committed_mix "$mix")
        run "$tag" "${BASE[@]}" --draft_threshold "$dt" --draft_fix_threshold "$dft" \
            --draft_top_k "$k" "${MIXARG[@]}"
      done
    done
  done
done

python evaluations/summarize_baseline.py --dir "$OUT"
echo "=== [$(date +%H:%M:%S)] v2_sweep done -> $OUT/summary.tsv ==="
