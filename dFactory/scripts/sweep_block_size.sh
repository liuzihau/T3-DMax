#!/usr/bin/env bash
# A — block-size sweep: does a smaller decode block close the think<->talk gap?
#
# Runs t3d_topk_eval_gsm8k across {modes} x {block sizes} and tabulates acc + compute.
# The decisive read is the GAP: if think_only barely moves 32->8 but seed (talk) improves
# a lot, that's hypothesis (b) (talk has a SHORTER reliable window) -> retraining talk at a
# smaller block_size is justified. If both move together, it's just (a) (smaller blocks help
# everyone) and block-8 won't close the gap.
#
# Usage (from dFactory/, env or PYTHONPATH set as for the eval):
#   THINK=../DMax-Math-16B-moe-merge \
#   TALK=./t3_topk_stage2_onpolicy_outputs/checkpoints/<step>/hf_ckpt \
#   bash scripts/sweep_block_size.sh
#
# Knobs (env): LIMIT (default 50), MODES (default "think_only seed"),
#              BLS (default "32 16 8"), EXTRA (extra eval flags, e.g. "--early_stop").
set -euo pipefail

THINK=${THINK:?set THINK=/path/to/think}
TALK=${TALK:?set TALK=/path/to/talk_ckpt}
LIMIT=${LIMIT:-50}
MODES=${MODES:-"think_only seed"}
BLS=${BLS:-"32 16 8"}
EXTRA=${EXTRA:-}

printf '%-14s | %5s | %6s | %8s | %7s | %11s\n' mode block acc think/ex talk/ex 20Lequiv/ex
printf '%s\n' "-------------------------------------------------------------------------"
for M in $MODES; do
  for BL in $BLS; do
    OUT=$(python -m tasks.t3d_topk_eval_gsm8k \
            --think_path "$THINK" --talk_path "$TALK" \
            --decode_mode "$M" --block_length "$BL" --limit "$LIMIT" $EXTRA 2>/dev/null)
    ACC=$(printf '%s\n' "$OUT" | grep -oP 'acc = \S+ = \K[0-9.]+'   || echo NA)
    TH=$( printf '%s\n' "$OUT" | grep -oP 'think/ex=\K[0-9.]+'       || echo NA)
    TK=$( printf '%s\n' "$OUT" | grep -oP 'talk/ex=\K[0-9.]+'        || echo NA)
    EQ=$( printf '%s\n' "$OUT" | grep -oP '20L-equiv/ex=\K[0-9.]+'   || echo NA)
    printf '%-14s | %5s | %6s | %8s | %7s | %11s\n' "$M" "$BL" "$ACC" "$TH" "$TK" "$EQ"
  done
done
