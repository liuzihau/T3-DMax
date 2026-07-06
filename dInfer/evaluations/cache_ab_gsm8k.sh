#!/usr/bin/env bash
# Copyright 2026 University of Sydney. Apache-2.0.
#
# QUICK cache A/B: one config (h0.9 d0.9 k2, gen512), N examples, decoded BOTH ways (no-cache vs the incremental
# cross-block prefix-KV cache), then acc + speed side by side. The cache is a valid DIFFERENT bf16 trajectory
# (proven correct in fp32), so JUDGE BY ACCURACY (iso) + MEAN speed, not token-identity.
#   DRAFTER=/abs/hf_ckpt HEAVY=/abs/DMax bash evaluations/cache_ab_gsm8k.sh
# Override via env: LIMIT (30), GEN (512), HTHR/DTHR/DK (0.9/0.9/2), OUT.
set -u
cd "$(dirname "$0")/.."                           # -> dInfer/

DRAFTER="${DRAFTER:-../dFactory/dbet_outputs/checkpoints/global_step_30000-v1/hf_ckpt/}"
HEAVY="${HEAVY:-../DMax-Math-16B-moe-merge}"
LIMIT="${LIMIT:-30}"
GEN="${GEN:-512}"
BLOCK="${BLOCK:-32}"
HTHR="${HTHR:-0.9}"; DTHR="${DTHR:-0.9}"; DK="${DK:-2}"
OUT="${OUT:-./cache_ab}"
mkdir -p "$OUT"
echo "[ab] drafter=$DRAFTER heavy=$HEAVY  limit=$LIMIT gen=$GEN  h$HTHR d$DTHR k$DK  -> $OUT"

# stats straight from the preds jsonl (robust to resume-skip): acc heavy/ex draft/ex wall/ex tok/s
report () {                                        # $1=tag $2=preds
  local tag="$1" preds="$2"
  local acc; acc=$(python evaluations/val_gsm8k.py --pred-path "$preds" --limit "$LIMIT" 2>/dev/null \
                   | grep -oE "Accuracy: [0-9.]+%" | tail -1 | grep -oE "[0-9.]+%")
  read heavy draft wall tps <<< "$(python3 -c "
import json
rows=[json.loads(l) for l in open('$preds')]
n=max(len(rows),1); tt=sum(r.get('wall_time',0.0) for r in rows) or 1e-9
print(f\"{sum(r.get('heavy_forwards',0) for r in rows)/n:.1f} {sum(r.get('draft_forwards',0) for r in rows)/n:.1f} {tt/n:.2f} {sum(r.get('gen_tokens',0) for r in rows)/tt:.1f}\")
" 2>/dev/null)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$tag" "${acc:-NA}" "${heavy:-NA}" "${draft:-NA}" "${wall:-NA}" "${tps:-NA}"
}

run () {                                           # $1=tag $2=extra_flag
  local tag="$1" extra="$2"
  local preds="$OUT/preds_${tag}.jsonl"
  echo; echo "==================== $tag ===================="
  if [ -s "$preds" ]; then echo "[ab] $preds exists -> skip"; else
    python evaluations/eval_dbet_gsm8k.py --drafter_path "$DRAFTER" --heavy_path "$HEAVY" \
      --out_path "$preds" --limit "$LIMIT" --gen_length "$GEN" --block_length "$BLOCK" \
      --heavy_threshold "$HTHR" --draft_threshold "$DTHR" --draft_top_k "$DK" $extra 2>&1 | tee "$OUT/gen_${tag}.log"
  fi
}

run nocache ""
run cache   "--use_cache"

echo; echo "======================= CACHE A/B (h$HTHR d$DTHR k$DK, n=$LIMIT) ======================="
{ printf "config\taccuracy\theavy/ex\tdraft/ex\twall/ex\ttok/s\n"
  report nocache "$OUT/preds_nocache.jsonl"
  report cache   "$OUT/preds_cache.jsonl"; } | column -t -s $'\t'
echo "======================================================================================"
echo "[ab] iso accuracy + lower cache wall/ex (mean) = win. logs/preds in $OUT/"
