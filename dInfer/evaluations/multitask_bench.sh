#!/bin/bash
# FINAL multi-task table: {gsm8k, asdiv, math500, algebra} x [ DMax {h0.3,h0.5} + DBet 4 settings x {h0.3,h0.5} ]
# = 40 eager runs. DBet settings = the four survivors of the endgame (§endgame): v1.5 ckpt, NOMARKOV, k2,
# fix 0.9, refine off, force default:  dt{0.5,0.6} x mix{hard,conf}.
#
# gen_length: 512 (gsm8k/asdiv), 1024 (math500/algebra -- Minerva solutions run long; 512 truncates).
# Rough wall (full sets, eager): gsm8k ~5h, asdiv ~5h, math500 ~4h, algebra ~9h => ~23h TOTAL.
# The script is RESUMABLE (skips existing jsonls) -- run it across nights, or set per-task limits:
#   [L_GSM8K=] [L_ASDIV=1000] [L_MATH500=] [L_ALGEBRA=500] bash evaluations/multitask_bench.sh
# Order: asdiv -> math500 -> algebra -> gsm8k (new information first; gsm8k largely confirms known rows).
# summarize_baseline picks each file's grader automatically from the rows' "task" field.

cd "$(dirname "$0")/.." || exit 1

HEAVY_EAGER=${HEAVY_EAGER:-../DMax-Math-16B-moe-merge}
DR=${DR:-../dFactory/dbet_v15_outputs/checkpoints/global_step_30000/hf_ckpt}
OUT=${OUT:-report_multitask_$(date +%m%d)}
mkdir -p "$OUT"

run() {  # run <tag> <limit> <cmd...>
  local tag=$1 lim=$2; shift 2
  if [ -s "$OUT/$tag.jsonl" ]; then echo "=== [skip] $tag ==="; return; fi
  local LIM=(); [ -n "$lim" ] && LIM=(--limit "$lim")
  echo "=== [$(date +%H:%M:%S)] $tag ==="
  if ! "$@" --out_path "$OUT/$tag.jsonl" "${LIM[@]}" >"$OUT/$tag.log" 2>&1; then
    [ -e "$OUT/$tag.jsonl" ] && mv "$OUT/$tag.jsonl" "$OUT/$tag.jsonl.failed"
    echo "!!! $tag FAILED (see $OUT/$tag.log)"
  fi
}

# task -> gen_length + limit env
gl()  { case $1 in math500|algebra) echo 1024;; *) echo 512;; esac; }
lim() { case $1 in gsm8k) echo "${L_GSM8K:-}";; asdiv) echo "${L_ASDIV:-}";;
                   math500) echo "${L_MATH500:-}";; algebra) echo "${L_ALGEBRA:-}";; esac; }

for task in asdiv math500 algebra gsm8k; do
  GL=$(gl "$task"); LM=$(lim "$task")
  BASE=(python evaluations/eval_dbet_gsm8k.py --task "$task" --heavy_path "$HEAVY_EAGER" \
        --drafter_path "$DR" --gen_length "$GL" --block_length 32)

  for h in 0.3 0.5; do
    # DMax control
    run "${task}_dmax_h${h}" "$LM" "${BASE[@]}" --heavy_threshold "$h" --heavy_only
    # DBet: the 4 endgame survivors (nomarkov, k2, fix 0.9)
    for dt in 0.5 0.6; do
      for mix in hard conf; do
        MIXARG=(); [ "$mix" = "conf" ] && MIXARG=(--draft_committed_mix conf)
        run "${task}_dbet_h${h}_dt${dt}_${mix}" "$LM" "${BASE[@]}" --heavy_threshold "$h" \
            --draft_threshold "$dt" --draft_fix_threshold 0.9 --draft_top_k 2 --no_markov "${MIXARG[@]}"
      done
    done
  done
done

python evaluations/summarize_baseline.py --dir "$OUT"
echo "=== [$(date +%H:%M:%S)] multitask_bench done -> $OUT/summary.tsv ==="
