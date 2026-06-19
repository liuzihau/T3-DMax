# T3-DMax (T3-D)

Think-Then-Talk variant built on LLaDA-2.0-mini and DMax's On-Policy Uniform Training (OPUT)
pipeline. Heavy think backbone produces a single per-block anchor; a lightweight 2-layer talk
model performs all iterative denoising steps inside the block.

See [`T3_DMax_implementation_brief.md`](./T3_DMax_implementation_brief.md) for the design.

## Layout

```
T3-D/
├── T3_DMax_implementation_brief.md       implementation brief (design + decisions)
├── README.md                              this file
├── NOTICE                                 attribution
├── dFactory/
│   ├── models/think_talk_llada2/         the new model
│   ├── tasks/train_t3_dmax_bd_oput.py    training entry point (talk-only OPUT rollout)
│   └── configs/sft/                       YAML configs
└── tests/                                 anchor-leak verification, smoke tests
```

When merging this into the cloned DMax repo (`git clone --recursive
https://github.com/czg1225/DMax.git T3-DMax`), copy the contents of `dFactory/` into the
clone's `dFactory/`, keeping DMax's `train.sh`, `scripts/`, `VeOmni/`, and existing models
intact. The new training task and config live alongside DMax's originals.

## Quickstart (after merging into DMax fork)

```bash
# 1. Build dataset (DMax's existing script — unchanged)
cd dFactory
python scripts/build_dataset_oput.py \
  --dataset_path Zigeng/DMax-LLaDA-2.0-Mini-Math-Trajectories \
  --out_dir ./my_data \
  --seed 42

# 2. Convert LLaDA-2.0-mini weights to merged MoE format (DMax's script)
python scripts/download_hf_model.py \
  --repo_id inclusionAI/LLaDA2.0-mini \
  --local_dir /path/to/separate_expert_model
python scripts/moe_convertor.py \
  --input-path /path/to/separate_expert_model \
  --output-path /path/to/LLaDA2.0-mini-moe-merge \
  --mode merge

# 3. Run anchor-leak verification (mandatory before training; brief §8.4)
pytest tests/test_anchor_leak.py

# 4. Launch training -- pick the config matching your GPU count

# Single H200 (141GB) -- freezes the think backbone (brief ablation A3)
PYTHONPATH=$(pwd)/VeOmni:$PYTHONPATH \
sh train.sh \
  tasks/train_t3_dmax_bd_oput.py \
  configs/sft/t3_llada2_mini_bd_oput_1gpu.yaml

# 2+ H200 -- full fine-tuning, FSDP2 sharded (strict A1 baseline)
# Edit train.sh's NPROC_PER_NODE or rely on `nvidia-smi --list-gpus | wc -l`.
PYTHONPATH=$(pwd)/VeOmni:$PYTHONPATH \
sh train.sh \
  tasks/train_t3_dmax_bd_oput.py \
  configs/sft/t3_llada2_mini_bd_oput_2gpu.yaml
```

| YAML | Distributed | Init | Trainable | model_config | ETA | Notes |
|---|---|---|---|---|---|---|
| `..._1gpu_smoke.yaml` | DDP | `cuda` | talk + LM head | `..._frozen_think` | ~10 min | `max_steps=500`, `lr=1e-4`. Pipeline validation only — don't keep the checkpoint. |
| `..._1gpu.yaml` | DDP, no-op | `cuda` | **talk only** | `..._talk_only` | ~4.4 days | Strategy A: think + LM head frozen, talk trains at `lr=1e-4`. Tests whether talk alone can match a frozen LLaDA-2.0-mini LM head. ~200M trainable params. |
| `..._2gpu.yaml` | FSDP2, full shard | `meta` | full (~16B) | `..._mini` | TBD | Strict A1 baseline. Needs 2+ H200. |

`model_config` references the directory under `dFactory/configs/model_configs/`:

- `think_talk_llada2_mini/` — `train_think=true`, `train_talk=true`, `train_lm_head=true` (full FT, used by 2gpu yaml).
- `think_talk_llada2_mini_frozen_think/` — `train_think=false`, `train_talk=true`, `train_lm_head=true` (talk + LM head trained; used by the smoke yaml for pipeline validation).
- `think_talk_llada2_mini_talk_only/` — `train_think=false`, `train_talk=true`, `train_lm_head=false` (talk only; the Strategy A real run on a single GPU).

All three yamls use the same data, optimizer, mask ratio, and OPUT rollout. The only
knobs that differ are the ones forced by distributed strategy + memory budget.

## Smoke test (recommended before committing to a real run)

```bash
# From the dFactory dir, with the conda env active and PYTHONPATH set:
cd dFactory
PYTHONPATH=$(pwd)/VeOmni:$PYTHONPATH \
sh train.sh \
  tasks/train_t3_dmax_bd_oput.py \
  configs/sft/t3_llada2_mini_bd_oput_1gpu_smoke.yaml \
  2>&1 | tee smoke_log.txt

# After the run finishes (or you Ctrl-C past step 200), validate:
python ../tools/check_smoke.py smoke_log.txt
```

`tools/check_smoke.py` reports PASS if:

- At least 100 steps were logged (no early crash).
- First-50-step loss mean is at log(vocab) ≈ 11.97 (sane initialisation).
- Last-50-step loss mean is below 11.0 (talk model is actually learning).
- Loss descends overall (final-50 mean < first-50 mean).
- No NaN/inf in loss or grad_norm.
- The mid-run checkpoint save line appears.

If PASS: the pipeline is end-to-end correct and you can either commit to the long
1gpu run, or request multi-GPU and switch to `..._2gpu.yaml`.

If FAIL: paste the report and the smoke log; the failure mode usually points at one
specific layer of the stack (data, model init, optimizer, checkpoint).

## Inference / GSM8K evaluation

End-to-end decode + GSM8K scoring lives in
[`dFactory/tasks/t3d_topk_eval_gsm8k.py`](./dFactory/tasks/t3d_topk_eval_gsm8k.py). It loads a
frozen **think** (full DMax) and a trained **talk** checkpoint, decodes the whole response
block-by-block, and reports accuracy + compute (`think/ex`, `talk/ex`, and the 20-layer-equivalent
`20L-equiv/ex = think_fwd + 0.5*talk_fwd`). Baseline to beat: **84% @ gen512**.

```bash
cd dFactory
export THINK=../DMax-Math-16B-moe-merge                                  # frozen full-DMax think
export TALK=./t3_topk_stage2_onpolicy_outputs/checkpoints/<step>/hf_ckpt # trained talk
#   (use ../merged_10L for the UNTRAINED talk = harness check + floor)
```

### Decode methods (`--decode_mode`)

Two families. In **(A)** think never commits — it only supplies per-position top-K candidates and
the **talk** does all committing. In **(B)** (ported from `t3d_probe_converged_teacher.mixed_converge`)
**whichever model runs a step commits** its own confident left-to-right prefix (DMax rule).

| `--decode_mode` | family | who commits | think cost / block | notes |
|---|---|---|---|---|
| `seed` (default) | A: think-as-candidate | talk only | `--seed_passes` (1 = once) | the H1 inference dynamic the training targets |
| `cross` | B: think-as-decoder | whoever runs | every other step | `think→talk→think→talk…` |
| `think_then_talk` | B: think-as-decoder | whoever runs | `--think_seed_count` leading commits | `think×N (commit) → talk…` |
| `cycle` | B: think-as-decoder | whoever runs | `--think_per_cycle : --talk_per_cycle` | repeating `(think×A, talk×B)`, e.g. `2,1` = `think,think,talk…` (think-heavy cross) |
| `think_only` | B: pure baseline | think | every step | pure DMax decode-to-converge baseline |

```bash
# 1. seed — H1 target: think once/block, talk drives (Stage-2 training dynamic)
python -m tasks.t3d_topk_eval_gsm8k --think_path $THINK --talk_path $TALK \
  --decode_mode seed --seed_passes 1 --gen_length 512 --block_length 32 \
  --threshold 0.3 --top_k 10 --limit 200

# 2. cross — think→talk→think→talk (strict alternation)
python -m tasks.t3d_topk_eval_gsm8k --think_path $THINK --talk_path $TALK \
  --decode_mode cross --limit 200

# 3. think_then_talk — think×2 commit, then talk to converge
python -m tasks.t3d_topk_eval_gsm8k --think_path $THINK --talk_path $TALK \
  --decode_mode think_then_talk --think_seed_count 2 --limit 200

# 4. cycle — think-heavy repeating schedule: think,think,talk,think,think,talk…
python -m tasks.t3d_topk_eval_gsm8k --think_path $THINK --talk_path $TALK \
  --decode_mode cycle --think_per_cycle 2 --talk_per_cycle 1 --limit 200

# 5. think_only — pure DMax baseline
python -m tasks.t3d_topk_eval_gsm8k --think_path $THINK --talk_path $TALK \
  --decode_mode think_only --limit 200
```

### Think-commit hand-off + soft-commit (`seed` mode)

Two `seed` knobs targeting the rollout collapse:
- **`--think_commit_threshold` (e.g. 0.6)**: think first COMMITS its own ≥-threshold confident prefix
  (no fallback) until it stalls, then talk takes only the uncertain tail. think handles the reliable
  bulk; talk is no longer starved on easy tokens. (`0` = legacy seed: think only seeds candidates.)
- **`--soft_commit`**: feed COMMITTED positions as DMax's soft top-K(+mask-residual) blend
  (`decode_uniform`'s committed=`soft_cond` behavior, `parallel_strategy.py:597,662`) instead of the
  hard token — keeps the committed region revisable so the candidate set can still shift across passes.

```bash
python -m tasks.t3d_topk_eval_gsm8k --think_path $THINK --talk_path $TALK \
  --decode_mode seed --think_commit_threshold 0.6 --soft_commit --trace --limit 50
```

### Early-stop (DMax termination) and other knobs

By default `seed` uses DMax's per-block Breakflag (all active ≥ 0.9 **or** no-change;
`parallel_strategy.py:578-590`); the mixed modes (`cross` / `think_then_talk` / `think_only`)
run each block to full convergence, capped at `--max_iters` (= `block_length` = 32). Add
**`--early_stop`** to apply DMax's full termination to *all* modes — the per-block 0.9 / no-change
gate **plus** the sequence-level EOS stop (stop generating once a block commits EOS,
`--eos_id 156892`; batch-filtering is a no-op at batch=1). Turn it on for compute that is directly
comparable across modes:

```bash
# Compute-fair sweep over all four modes (DMax-faithful termination)
for M in seed cross think_then_talk think_only; do
  python -m tasks.t3d_topk_eval_gsm8k --think_path $THINK --talk_path $TALK \
    --decode_mode $M --early_stop --eos_id 156892 --limit 200
done
```

Other flags: `--threshold` (commit confidence, DMax default 0.3; the probe uses 0.6),
`--top_k` (candidate set size, default 10), `--max_iters` (per-block cap, default 32),
`--no_mask_residual` (feed talk the no-mask top-K — match a talk trained with
`keep_mask_residual=false`), `--limit` (number of GSM8K test problems), `--debug_print`
(dump the full generation per example).

### Per-block diagnostic probe (no full decode)

[`dFactory/tasks/t3d_probe_converged_teacher.py`](./dFactory/tasks/t3d_probe_converged_teacher.py)
compares the strategies on two consecutive blocks of a few prompts, split into the committed
prefix vs the still-masked tail, with per-position confidence + per-method think/talk forward
counts and wall/model timing. Use it to see *where* a method diverges before paying for a full
eval run.

```bash
python -m tasks.t3d_probe_converged_teacher --think_path $THINK --talk_path $TALK \
  --block_length 32 --top_k 10 --threshold 0.6 --gen_block 1 --limit 5
```

### Why is talk worse than think? — block-window diagnostics

Two tools to test the hypothesis that **talk has a shorter reliable decode window than think**
(so a smaller `block_size` would close the gap), *before* committing to a retrain.

**A — block-size sweep** ([`scripts/sweep_block_size.sh`](./dFactory/scripts/sweep_block_size.sh)):
runs the eval across `{block size} x {mode}` and tabulates acc + compute. The decisive read is the
**gap** — if `think_only` barely moves 32→8 but `seed` (talk) improves a lot, talk has the shorter
window and retraining at a smaller block is justified; if both move together it's just the generic
quality/parallelism trade-off and block-8 won't help.

```bash
cd dFactory
THINK=$THINK TALK=$TALK LIMIT=50 MODES="think_only seed" BLS="32 16 8" \
  bash scripts/sweep_block_size.sh
```

**B — within-block-position profile**
([`dFactory/tasks/t3d_position_profile.py`](./dFactory/tasks/t3d_position_profile.py)): on a
fully-masked block, measures per within-block position `j` each model's single-forward confidence
and agreement with think's converged decode (the per-token truth proxy). Prints a per-position +
binned table and the **reliable window** (largest prefix with agreement ≥ `--window_floor`).
`talk window << think window` confirms the hypothesis mechanistically.

```bash
python -m tasks.t3d_position_profile --think_path $THINK --talk_path $TALK \
  --block_length 32 --top_k 10 --threshold 0.3 --gen_block 1 --limit 50
```

**Rollout-collapse trace** (`--trace`, `seed` mode only): when the talk self-rollout collapses
(`seed` → 0% / hundreds of talk passes), this instruments the decode to pin *which* failure it is.
Per within-block pass index it logs commits/pass, talk confidence, and **overlap of talk's commits
with think's seed top-K**; per block, passes-to-converge + adjacent-repeat fraction + cap-hit rate;
per sequence, no-EOS. Signatures: overlap falling = **coverage drift**; commits/pass ≈ 1 =
**commit starvation**; high repeat fraction = **soft-embed degeneracy**.

```bash
python -m tasks.t3d_topk_eval_gsm8k --think_path $THINK --talk_path $TALK \
  --decode_mode seed --seed_passes 1 --trace --limit 50
```

## Acknowledgements

This repository builds on two prior projects:

- **DMax: Aggressive Parallel Decoding for dLLMs** (Chen, Fang, Ma, Yu, Wang; NUS, 2026).
  <https://github.com/czg1225/DMax> — Apache-2.0.
  We reuse and adapt its `dFactory/` training pipeline, OPUT data processing, block-diffusion
  training scripts, and the `LLaDA2Moe*` modeling code. Files derived from DMax are marked
  in their headers with the original copyright.

- **Think-Then-Talk** (internal — University of Sydney).
  We vendor the talk-model architecture (gated residual anchor conditioning, RPS state
  update, two-layer transformer talk block) from this codebase. Files derived from
  Think-Then-Talk are marked in their headers.

We extend H1 in the brief (architecture viability — match DMax-comparable accuracy at lower
per-step inference compute by amortising the heavy backbone over each iteration of a block).

## License

Apache-2.0. See [`NOTICE`](./NOTICE) for required attributions to upstream projects.
