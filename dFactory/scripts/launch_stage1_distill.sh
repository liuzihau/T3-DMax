#!/usr/bin/env bash
# T3-D top-K talk — STAGE 1 cold start, WITH Path-A think->talk distillation.
#
# What's new vs the prior stage-1: Path A (the [MASK] path) no longer trains on one-hot
# gold alone — it forwards the frozen 16B think and adds beta * forward-KL(think||talk) at
# the predict positions, so think's dark knowledge (full top-K) trains the talk. Mass-
# covering => cold-start alignment to PARITY (not beating think; that's stage 2).
# Path B is UNCHANGED (gold-pure top-K selection = the H2 lever). Knobs live in the config:
#   t3_distill_beta / t3_distill_alpha / t3_distill_temp   (set beta=0 to revert to legacy).
#
# Run from the dFactory dir on the TRAINING box (needs GPU + models + data):
#   bash scripts/launch_stage1_distill.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."          # -> dFactory
CONFIG="configs/sft/t3_topk_talk_stage1_coldstart.yaml"

RED='\033[0;31m'; GRN='\033[0;32m'; RST='\033[0m'
ok()   { printf "${GRN}[ok]${RST}  %s\n" "$*"; }
die()  { printf "${RED}[missing]${RST} %s\n" "$*"; exit 1; }

echo "== stage-1 distill preflight =="
[ -f train.sh ]              && ok "train.sh"                  || die "train.sh"
[ -d VeOmni ]               && ok "VeOmni"                    || die "VeOmni"
[ -f "$CONFIG" ]            && ok "$CONFIG"                   || die "$CONFIG"
[ -f my_data/postprocess_train.jsonl ] && ok "train jsonl"   || die "my_data/postprocess_train.jsonl (run scripts/build_dataset_oput.py first)"
[ -d ../merged_10L ]        && ok "talk init (merged_10L)"    || die "../merged_10L"
[ -d ../DMax-Math-16B-moe-merge ] && ok "think model"        || die "../DMax-Math-16B-moe-merge"

echo "== distill config in effect =="
grep -E "t3_distill_(beta|alpha|temp)" "$CONFIG" || true
echo

# 1-GPU pattern (DDP + init_device:cuda + no full-shard); train.sh auto-detects GPU count.
PYTHONPATH="$(pwd)/VeOmni:${PYTHONPATH:-}" \
  sh train.sh tasks/train_t3_topk_talk.py "$CONFIG"
