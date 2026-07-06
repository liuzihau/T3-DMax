# evaluations/attic — parked diagnostics & superseded scripts

Moved here to declutter `evaluations/` once the DBet prefix-KV **cache investigation closed**
(verdict: cache is *correct* but only a marginal win at batch=1, which is memory-bandwidth-bound;
kept as an off-by-default `--use_cache` flag). These still run (paths were fixed for the deeper
folder); resurrect if we revisit caching or need the correctness proofs again.

## Cache-correctness diagnostics (reusable)
Run from the `dInfer/` root, e.g. `bash evaluations/attic/prove_cache.sh`, or
`python evaluations/attic/diag_cache.py ...`.

- **`prove_cache.sh`** — one-shot battery: forward-level Δ (a/b/c) + decode-level bit-identity
  (1: bf16 `--exact_moe --math_attn` → 10/10; 2: fp32 `--exact_moe` → 10/10). The proof that the
  cache is *logically correct* and the bf16 token-divergence is shape-dependent rounding, not a bug.
- **`diag_cache.py`** — forward-level cache vs no-cache Δ, per-layer localization, `--prove_shape`
  (cache-free control showing the heavy's block output is shape-dependent in bf16), `--exact_moe`,
  `--math_attn`, `--cache_build {crop,separate}`.
- **`validate_cache.py`** — decode-level: run each prompt no-cache vs `--use_cache`, assert token-identical.
- **`diag_determinism.py`, `diag_attn.py`** — isolate determinism / prefix-length / MoE-routing vs
  attention as the source of cache non-exactness.

## Superseded
- **`sweep_dbet_gsm8k.sh`** — the original threshold sweep; replaced by
  `evaluations/sweep_dbet_gsm8k_indep.sh` (explicit config list, per-config grade/degen/throughput,
  `summary.tsv`, `USE_CACHE=1` toggle).

## Still live in `evaluations/` (NOT here)
`eval_dbet_gsm8k.py` (main eval, has `--use_cache`), `sweep_dbet_gsm8k_indep.sh` (main battery),
`cache_ab_gsm8k.sh` (quick ±cache A/B — also handy for the batched premise check).
