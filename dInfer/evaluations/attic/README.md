# evaluations/attic — parked diagnostics & superseded scripts

Archived to keep `evaluations/` to the current DBet pipeline. Nothing here is on the live path; resurrect if a
line of investigation reopens. See `../../README.md` for what IS live.

## Phase-3 SGLang milestones (validation done)
- **`validate_sglang_hsel.py`** — G1: proves the SGLang heavy's tapped h_sel/h_last match the eager heavy (hook tap
  AND `--buffer_tap` graph-safe tap). Verdict: ARGMAX_MATCH 98.4%, drafter tolerates the shift.
- **`eval_dbet_sglang_gsm8k.py`** — G2: the no-cache SGLang-DBet driver (iso-accurate 89%). Superseded by the
  cached+graph driver `evaluations/eval_dbet_sglang_cached_gsm8k.py`.
- **`bench_draft_forward.py`** — the draft-time kill-switch (compiled draft 1.47ms vs heavy 5.6ms → DBet has room).

## Cache-correctness diagnostics (the prefix-KV cache investigation)
Verdict: cache is *correct* but marginal at batch=1 (memory-bandwidth-bound). Run from the `dInfer/` root.
- **`prove_cache.sh`** — decode-level bit-identity battery (fp32 `--exact_moe` → 10/10 identical).
- **`diag_cache.py` / `diag_determinism.py` / `diag_attn.py`** — forward-level Δ, per-layer localization,
  determinism / prefix-length / MoE-vs-attention source isolation.
- **`validate_cache.py`** — decode-level no-cache vs `--use_cache` token-identity.
- **`cache_ab_gsm8k.sh`** — quick ±cache A/B on GSM8K.

## Superseded sweeps / other-branch
- **`sweep_dbet_gsm8k.sh`**, **`sweep_dbet_gsm8k_indep.sh`** — DBet threshold sweeps (per-config grade/degen/tps).
- **`sweep_dmax_gsm8k.sh`**, **`sweep_llada2_gsm8k.sh`** — heavy/LLaDA2 baseline sweeps.
- **`eval_dinfer_t3d.py`, `eval_t3d_gsm8k.py`, `eval_t3d_mini.sh`** — T3-DMax (other-branch) evals.
- **`probe*.py` / `probe*.sh`** — the E0/E1 layer-probe exploration (probe_runner era).
