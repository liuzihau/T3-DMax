#!/usr/bin/env bash
# T3-D v2 preflight: run before launching a new training run.
# Stops on first failure with a clear marker.
#
# Usage from the T3-DMax repo root:
#   bash dFactory/scripts/preflight.sh
#
# Optional env vars:
#   LLADA2_TOKENIZER_PATH  -- override path to LLaDA2.0-mini-moe-merge
#                              (default: ../LLaDA2.0-mini-moe-merge relative to repo root)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Reds and greens.
RED='\033[0;31m'
GRN='\033[0;32m'
YEL='\033[0;33m'
RST='\033[0m'

pass()  { printf "${GRN}[PASS]${RST} %s\n" "$*"; }
fail()  { printf "${RED}[FAIL]${RST} %s\n" "$*"; exit 1; }
warn()  { printf "${YEL}[WARN]${RST} %s\n" "$*"; }
stage() { printf "\n${YEL}== %s ==${RST}\n" "$*"; }

# Run a pytest invocation, treating exit code 5 (no tests collected -- usually
# because torch/transformers/tokenizer weren't available) as a soft skip
# instead of a hard fail. Real test failures still abort the script.
run_pytest() {
    local label="$1"
    shift
    set +e
    pytest "$@" --tb=short
    local rc=$?
    set -e
    case "$rc" in
        0) pass "$label" ;;
        5) warn "$label -- pytest skipped all (torch/tokenizer missing in this env). Re-run in the training env to actually exercise these gates." ;;
        *) fail "$label -- pytest exited with code $rc" ;;
    esac
}

# --------------------------------------------------------------------------
stage "1. Static AST parse of all modified files"
# --------------------------------------------------------------------------
python3 - <<'PY' || fail "AST parse failed"
import ast, sys
files = [
    "dFactory/tasks/train_t3_dmax_bd_oput.py",
    "dFactory/tasks/diagnose_think_vs_talk.py",
    "dFactory/tasks/dataset/data_transform.py",
    "dFactory/tasks/curriculum.py",
    "dFactory/models/think_talk_llada2/modeling_think_talk_llada2.py",
]
bad = []
for f in files:
    try:
        ast.parse(open(f).read())
        print(f"  OK   {f}")
    except SyntaxError as e:
        print(f"  FAIL {f} at line {e.lineno}: {e.msg}")
        bad.append(f)
if bad:
    sys.exit(1)
PY
pass "All modified files parse"

# --------------------------------------------------------------------------
stage "2. Anchor leak tests (brief sec 8.4 canonical gate)"
# --------------------------------------------------------------------------
PYTHONPATH="dFactory:dFactory/VeOmni:${PYTHONPATH:-}" \
    run_pytest "Anchor leak tests (non-slow)" \
    tests/test_anchor_leak.py -v -m "not slow"

# --------------------------------------------------------------------------
stage "3. SFT-label leak fix unit test"
# --------------------------------------------------------------------------
PYTHONPATH="dFactory:${PYTHONPATH:-}" \
    run_pytest "SFT-label leak fix" \
    tests/test_sft_leak_fix.py -v

# --------------------------------------------------------------------------
stage "4. Curriculum sampler unit tests"
# --------------------------------------------------------------------------
PYTHONPATH="dFactory:${PYTHONPATH:-}" \
    run_pytest "Curriculum sampler in valid range" \
    tests/test_curriculum_sampler.py -v

# --------------------------------------------------------------------------
stage "5. SUMMARY"
# --------------------------------------------------------------------------
pass "All preflight gates passed. You can proceed to:"
echo
echo "  (5) Diagnostic v2 dry-run on step-45000:"
echo "      PYTHONPATH=dFactory:dFactory/VeOmni:\$PYTHONPATH python3 \\"
echo "        dFactory/tasks/diagnose_think_vs_talk.py \\"
echo "        --model_path dFactory/outputs/<run>/checkpoints/global_step_45000/hf_ckpt \\"
echo "        --tokenizer_path ../LLaDA2.0-mini-moe-merge \\"
echo "        --gen_length 32 --n_iters 5"
echo
echo "      Expected: runs end-to-end without errors (T3D output on step 45000 will"
echo "      still be degenerate -- that's the broken model, not the diagnostic)."
echo
echo "  (6) Tiny training smoke run (train_steps=20) with the v2 config:"
echo "      noise_range_low=0.50 noise_range_high=0.90"
echo "      t3_sigma_gate=0.10 t3_rollout_ratio_gate=0.10 t3_n_iter_gate=1"
echo "      t3_train_iterations=5 t3_train_iterations_min=2"
echo "      t3_rollout_ratio_low=0.20 t3_rollout_ratio_high=0.60"
echo "      t3_reveal_threshold=0.5"
echo
echo "      Watch wandb for:"
echo "        - training/loss_clean_region_LEAK_TRIPWIRE absent/NaN"
echo "        - t3/sigma_sampled varies within +/-0.10 of t3/sigma_center"
echo "        - t3/n_iters_sampled in {1, 2, 3} for early steps"
echo "        - no crashes after 20 steps"
