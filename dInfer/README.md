# dInfer — DBet inference & evaluation

Runs the **DBet drafter under DMax's optimized SGLang decode**: the frozen 16B heavy runs in SGLang (prefix KV
cache + CUDA graphs + compile — the ~1000 tok/s batch=1 path), and the trained drafter is injected into the
block-diffusion loop to commit extra tokens, cutting the number of heavy forwards. Training lives in
[`../dFactory`](../dFactory/README.md).

## Environment
Use the `dInfer` conda env (sglang 0.5.3.post1). The heavy loads from the **per-expert** `../DMax-Math-16B`
(the loader fuses experts; the pre-merged checkpoint fails to load); the drafter's frozen embed/lm_head/norm load
from the same dir. bf16 only.

## Evaluate DBet (the main path)
```bash
# graphs + compiled draft = the optimized run; grade after
python evaluations/eval_dbet_sglang_cached_gsm8k.py \
    --heavy_path ../DMax-Math-16B \
    --drafter_path ../dFactory/dbet_outputs/checkpoints/global_step_30000-v1/hf_ckpt \
    --out_path preds.jsonl --limit 100 --cuda_graph --compile_draft
python evaluations/val_gsm8k.py --pred-path preds.jsonl --limit 100
```
Reports GSM8K accuracy (after grading) + tok/s + `fwd/ex` (heavy forwards) + `draft_commits/ex`.

- Debug the injection first with **graphs OFF** (drop `--cuda_graph --compile_draft`): should be ~iso-accurate and
  `draft_commits/ex > 0` (if `0`, the drafter isn't being hit).
- **Heavy-only baseline** in the *same* decode: add `--no_draft` (a real off-switch — `--draft_threshold` can't
  disable the drafter, since EXTEND always commits the leftmost slot for progress). Compare its `fwd/ex` to DBet's.
- Standalone heavy tok/s: `evaluations/bench_sglang_heavy_tps.sh`.
- Eager (no SGLang) DBet reference: `evaluations/eval_dbet_gsm8k.py`.

## How it works (Option A: heavy-in-SGLang + injected drafter)
| path | what |
|---|---|
| `python/dinfer/model/modeling_llada2_moe_sglang_dbet.py` | **copy** of the SGLang heavy + a graph-safe buffer tap (`enable_dbet_tap`/`pop_dbet_features`): writes h_sel/h_last via `copy_` inside `forward()` so features survive CUDA-graph replay (Python hooks don't). Original file untouched. |
| `python/dinfer/decoding/generate_dbet_sglang_cached.py` | `DbetBlockDiffusionIteration`/`DbetBlockDiffusionLLM` — injects the drafter into DMax's `forward_uniform`: after the heavy commit, pop the tap, run the drafter (canvas + cross-block draft prefix cache), EXTEND/FIX into the token array + soft-embed into the next-step feed. |
| `python/dinfer/decoding/generate_dbet.py` | eager drafter decode + `load_drafter_standalone` (loads the ~350M drafter with frozen pieces from the heavy, no 2nd 16B). |
| `evaluations/eval_dbet_sglang_cached_gsm8k.py` | the driver above. **The tap must be enabled before the ModelRunner captures its CUDA graph** — the driver does this in `build()`. |
| `evaluations/eval_dinfer_sglang.py` | DMax's reference SGLang eval (this driver is derived from it). |
| `evaluations/val_*.py`, `evaluations/tasks/` | graders + lm-eval task specs. |
| `evaluations/attic/` | archived probes/validators/sweeps (feature-tap validation, cache A/B, G1/G2 milestones). |
