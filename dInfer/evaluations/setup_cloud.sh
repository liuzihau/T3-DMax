#!/usr/bin/env bash
# One-time setup for the lens-surface experiment on a fresh cloud GPU box (Lightning AI H100, 80GB).
# Follows the DMax README install flow, MINUS what our eager-only probe does not need:
# no second conda env, no sglang, no vllm (the decoding __init__ is try/except-wrapped for exactly this).
#
# Needs: ~150 GB disk (33 GB original + 33 GB merged weights + HF cache + shards), CUDA torch preinstalled
# (Lightning studio images ship it -- we keep it and do NOT let pip replace it).
#
# Usage:
#   bash setup_cloud.sh              # clone + deps + model download/merge + import self-check
#   REPO_DIR=... MODEL_DIR=... bash setup_cloud.sh   # override locations
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/T3-DMax}"
MODEL_DIR="${MODEL_DIR:-$HOME/models}"
REPO_URL="${REPO_URL:-https://github.com/liuzihau/T3-DMax.git}"
HF_REPO="${HF_REPO:-Zigeng/DMax-Math-16B}"
ORIG="$MODEL_DIR/$(basename "$HF_REPO")"
MERGED="${ORIG}-moe-merge"

echo "=== [1/5] clone (DARC branch, with the VeOmni submodule) ==="
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --recursive -b DARC "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"
git submodule update --init dFactory/VeOmni

echo "=== [2/5] python deps (keep the studio's torch) ==="
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA torch in this env'; \
print('torch', torch.__version__, 'cuda', torch.version.cuda)"
# VeOmni provides the fused-MoE kernel + pulls transformers/datasets/tiktoken/etc.
# If the resolver tries to touch torch, abort and re-run with: pip install -e dFactory/VeOmni --no-deps
# followed by: pip install "transformers<4.57" datasets tiktoken torchdata blobfile psutil timm
pip install -e dFactory/VeOmni
pip install numpy pandas matplotlib sentencepiece protobuf "huggingface_hub[hf_transfer]"
python -c "import torch; assert torch.cuda.is_available(), \
'pip replaced torch with a CPU build -- reinstall the studio torch, then use the --no-deps route above'"

echo "=== [3/5] download original weights ($HF_REPO) ==="
mkdir -p "$MODEL_DIR"
HF_HUB_ENABLE_HF_TRANSFER=1 python dFactory/scripts/download_hf_model.py \
  --repo_id "$HF_REPO" --local_dir "$ORIG"

echo "=== [4/5] merge experts (required by the probe's fused load path) ==="
if [ ! -d "$MERGED" ]; then
  python dFactory/scripts/moe_convertor.py -i "$ORIG" -o "$MERGED" -m merge
  # the convertor may not copy tokenizer/config sidecars -- fill in whatever is missing
  for f in "$ORIG"/*.json "$ORIG"/*.model "$ORIG"/*.txt; do
    [ -e "$f" ] || continue
    base="$(basename "$f")"
    [ -e "$MERGED/$base" ] || cp "$f" "$MERGED/"
  done
fi

echo "=== [5/5] import self-check (fused kernel + probe imports, no forward) ==="
cd "$REPO_DIR/dInfer/evaluations"
python - <<'EOF'
import os, sys
here = os.getcwd()
for p in (os.path.abspath("../python"), os.path.abspath("../../dFactory"),
          os.path.abspath("../../dFactory/VeOmni")):
    sys.path.insert(0, p)
from dinfer.decoding.generate_t3d import build_block_causal_mask, dmax_commit_uniform, _soft_embed  # noqa
from models.llada2_moe import modeling_llada2_moe as m
assert m.fused_moe_forward is not None, "fused-MoE kernel NOT importable -- check the VeOmni install"
print("[self-check] probe imports OK; fused-MoE kernel available")
EOF

echo ""
echo "Setup done. Next:"
echo "  cd $REPO_DIR/dInfer/evaluations"
echo "  bash run_lens_surface.sh smoke $MERGED     # ~15 min; eyeball runs/lens_smoke/analysis"
echo "  bash run_lens_surface.sh full  $MERGED     # few hours; then download runs/lens_surface + STOP the GPU"
