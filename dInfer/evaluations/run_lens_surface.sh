#!/usr/bin/env bash
# Lens-surface experiment launcher (arc_head/exp_lens_surface.md, trial 1).
# Usage:
#   bash run_lens_surface.sh smoke   <merged_ckpt_path>     # ~4 samples, minutes -- run this FIRST
#   bash run_lens_surface.sh full    <merged_ckpt_path>     # 150 samples, hours
#   bash run_lens_surface.sh analyze <run_dir> [tokenizer]  # offline, CPU is fine
#   bash run_lens_surface.sh dash <run_dir> <sample> <block> [tokenizer]   # per-block zoom-in dashboard
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:?smoke|full|analyze}"

case "$MODE" in
  smoke)
    CKPT="${2:?merged checkpoint path}"
    python probe_lens_surface.py --model_path "$CKPT" \
      --out_dir runs/lens_smoke --limit 4 --gen_length 128 --threshold 0.5 --shard_samples 4
    python probe_lens_analyze.py --run_dir runs/lens_smoke --tokenizer_path "$CKPT"
    echo "smoke OK -> runs/lens_smoke/analysis (check fig1 + REPORT.md before the full run)"
    ;;
  full)
    CKPT="${2:?merged checkpoint path}"
    python probe_lens_surface.py --model_path "$CKPT" \
      --out_dir runs/lens_surface --limit 150 --gen_length 256 --threshold 0.5
    echo "recording done -> runs/lens_surface (analyze on CPU: bash run_lens_surface.sh analyze runs/lens_surface $CKPT)"
    ;;
  analyze)
    RUN="${2:?run dir with shards}"
    TOK="${3:-}"
    if [ -n "$TOK" ]; then
      python probe_lens_analyze.py --run_dir "$RUN" --tokenizer_path "$TOK"
    else
      python probe_lens_analyze.py --run_dir "$RUN"
    fi
    ;;
  dash)
    RUN="${2:?run dir with shards}"
    S="${3:?sample index}"
    B="${4:?block index}"
    TOK="${5:-}"
    ARGS=(--run_dir "$RUN" --sample "$S" --block "$B" --steps 1,3,5,-1 --layers 1: --topk 5)
    [ -n "$TOK" ] && ARGS+=(--tokenizer_path "$TOK")
    python probe_lens_dashboard.py "${ARGS[@]}"
    ;;
  *) echo "unknown mode: $MODE (smoke|full|analyze|dash)"; exit 1 ;;
esac
