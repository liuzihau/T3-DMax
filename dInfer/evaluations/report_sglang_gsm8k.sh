#!/usr/bin/env bash
# Report: LLaDA-2.0, DMax-Math-16B, and DBet on GSM8K (N=200) under DMax's SGLang cached + CUDA-graph decode.
# Accuracy (val_gsm8k) + throughput (tps = tokens/sec, tpf = tokens/heavy-forward). ONE driver for all three so the
# comparison is fair: eval_dbet_sglang_cached_gsm8k.py with --no_draft is the *identical* heavy decode (== DMax's
# BlockDiffusionLLM) pointed at any --heavy_path; with the drafter it's DBet. Same prompt, same grader.
#
#   LLADA_PATH=../LLaDA2.0-mini DMAX_PATH=../DMax-Math-16B \
#   DRAFTER_PATH=../dFactory/dbet_outputs/checkpoints/global_step_30000-v1/hf_ckpt \
#   bash evaluations/report_sglang_gsm8k.sh
#
# Runs (8): LLaDA {th0.5,0.7,0.9} | DMax {th0.5,0.7,0.9} | DBet {th0.9 k2, th0.9 k3}. ~1.5h (each reloads the 16B).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE/.."          # -> dInfer/
export PYTHONPATH="$HERE/../python:${PYTHONPATH:-}"

LLADA_PATH="${LLADA_PATH:-../LLaDA2.0-mini}"
DMAX_PATH="${DMAX_PATH:-../DMax-Math-16B}"
DRAFTER_PATH="${DRAFTER_PATH:-../dFactory/dbet_outputs/checkpoints/global_step_30000-v1/hf_ckpt}"
N="${N:-200}"; GEN="${GEN:-512}"; BLOCK="${BLOCK:-32}"
OUT="${OUT:-report_sglang_gsm8k}"; mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
DRIVER=evaluations/eval_dbet_sglang_cached_gsm8k.py
GRADER=evaluations/val_gsm8k.py

[ -e "$LLADA_PATH" ] || echo "WARN: LLADA_PATH=$LLADA_PATH not found (set LLADA_PATH=...)"
[ -e "$DMAX_PATH" ]  || echo "WARN: DMAX_PATH=$DMAX_PATH not found"
printf 'system\tconfig\tacc\ttps\ttpf\tfwd/ex\tn\n' > "$SUMMARY"

run_one () {   # $1=system  $2=config-label  $3=heavy_path  $4..=extra driver args
  local sys="$1" cfg="$2" hp="$3"; shift 3
  local tag="${sys}_${cfg}" preds="$OUT/preds_${sys}_${cfg}.jsonl" log="$OUT/log_${sys}_${cfg}.txt"
  echo "=== [$sys $cfg]  ($N ex, gen=$GEN) ==="
  if ! python "$DRIVER" --heavy_path "$hp" --out_path "$preds" --limit "$N" \
        --gen_length "$GEN" --block_length "$BLOCK" "$@" > "$log" 2>&1; then
    echo "  FAILED -> $log"; tail -6 "$log"
    printf '%s\t%s\tFAIL\t-\t-\t-\t%s\n' "$sys" "$cfg" "$N" >> "$SUMMARY"; return
  fi
  local acc; acc=$(python "$GRADER" --pred-path "$preds" --limit "$N" 2>/dev/null | awk -F': ' '/Accuracy:/{print $2}')
  read -r tps tpf fwd < <(python - "$preds" <<'PY'
import json, sys
tok = w = f = n = 0
for line in open(sys.argv[1]):
    r = json.loads(line); tok += r.get("gen_tokens", 0); w += r.get("wall_time", 0.0)
    f += r.get("forwards", 0); n += 1
n = max(n, 1)
print(f"{tok/max(w,1e-9):.1f} {tok/max(f,1):.2f} {f/n:.1f}")
PY
)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$sys" "$cfg" "${acc:-NA}" "$tps" "$tpf" "$fwd" "$N" >> "$SUMMARY"
  echo "  acc=${acc:-NA}  tps=${tps}  tpf=${tpf}  fwd/ex=${fwd}"
}

# --- backbones: heavy-only threshold sweep (identical decode, drafter off) ---
for T in 0.5 0.7 0.9; do run_one LLaDA "th${T}" "$LLADA_PATH" --no_draft --heavy_threshold "$T" --cuda_graph; done
for T in 0.5 0.7 0.9; do run_one DMax  "th${T}" "$DMAX_PATH"  --no_draft --heavy_threshold "$T" --cuda_graph; done
# --- DBet: heavy th0.9 + draft th0.9, top-k 2 & 3 (compiled draft) ---
for K in 2 3; do
  run_one DBet "th0.9_k${K}" "$DMAX_PATH" --drafter_path "$DRAFTER_PATH" \
      --heavy_threshold 0.9 --draft_threshold 0.9 --draft_top_k "$K" --cuda_graph --compile_draft
done

echo ""; echo "===== REPORT: GSM8K N=$N, SGLang cached+graph (gen=$GEN, block=$BLOCK) ====="
awk -F'\t' '{printf "%-7s %-10s %-9s %-9s %-7s %-7s %-4s\n",$1,$2,$3,$4,$5,$6,$7}' "$SUMMARY"
echo "(tps=tokens/s, tpf=tokens/heavy-forward; DBet fwd/ex counts heavy forwards only). Full: $SUMMARY"
