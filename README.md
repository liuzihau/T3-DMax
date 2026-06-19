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

Three methods, all in one unified decoder (`decode_t3d`). A per-block schedule picks think|talk each
step; **whichever model runs commits** its own confident left-to-right prefix (the single DMax
threshold rule, + EOS). Per-forward input is fixed — no flags: **committed** positions always get the
DMax soft top-K(+mask-residual) blend of the latest logits; **masked** positions get bare `[MASK]` for
a think step, or the top-K soft blend (+mask residual) of the previous pass's logits for a talk step.

| `--decode_mode` | schedule | knob |
|---|---|---|
| `base_think` | `T T T T …` — think only (DMax baseline) | — |
| `think_then_talk` | `T×a` then `t t t …` — think commits `a`, talk takes over to converge | `--think_passes a` |
| `interleave` | repeating `T×t, t` — e.g. `t=3` → `TTTt TTTt …` (`t=1` = strict alternation) | `--think_per_talk t` |

```bash
# base think — pure DMax baseline
python -m tasks.t3d_topk_eval_gsm8k --think_path $THINK --talk_path $TALK \
  --decode_mode base_think --gen_length 512 --block_length 32 --threshold 0.3 --limit 200

# think_then_talk — think commits 1 pass, then talk to converge
python -m tasks.t3d_topk_eval_gsm8k --think_path $THINK --talk_path $TALK \
  --decode_mode think_then_talk --think_passes 1 --limit 200

# interleave — think think think talk, repeating
python -m tasks.t3d_topk_eval_gsm8k --think_path $THINK --talk_path $TALK \
  --decode_mode interleave --think_per_talk 3 --limit 200
```

### Early-stop and other knobs

By default each block runs to full convergence (capped at `--max_iters` = `block_length`). Add
**`--early_stop`** for DMax's termination — per-block Breakflag (all active ≥ 0.9 or no-change,
`parallel_strategy.py:578-590`) plus the sequence-level EOS stop (`--eos_id 156892`; batch-filtering
is a no-op at batch=1). Other flags: `--threshold` (commit confidence, default 0.3), `--top_k`
(candidate set, default 10), `--max_iters` (per-block cap, default 32), `--limit`, `--debug_print`.

### `--trace` (rollout diagnostics)

Instruments the decode: per within-block pass index it logs commits/pass, the deciding model's
confidence, and **overlap of commits with the iter-0 top-K** (overlap < 1 = fresh tokens entering the
pool); per block, passes-to-converge + adjacent-repeat fraction + cap-hit rate; per sequence, no-EOS.
On `base_think` it's the think confidence baseline (high/holding, converges in ~3–4 passes); on
`think_then_talk`/`interleave` it shows talk's per-pass confidence and whether think re-forwards inject
fresh candidates (overlap dropping).

```bash
python -m tasks.t3d_topk_eval_gsm8k --think_path $THINK --talk_path $TALK \
  --decode_mode base_think --trace --limit 50
```

### Diagnostics (per-block probe, position profile, block-size sweep)

- [`t3d_probe_converged_teacher.py`](./dFactory/tasks/t3d_probe_converged_teacher.py) — per-block
  comparison with a per-position confidence table (committed prefix vs still-masked tail).
- [`t3d_position_profile.py`](./dFactory/tasks/t3d_position_profile.py) — per within-block-position
  single-forward confidence + agreement-with-think-converged, and the **reliable window** readout.
- [`scripts/sweep_block_size.sh`](./dFactory/scripts/sweep_block_size.sh) — sweeps `--decode_mode ×
  block_size`, tabulating acc + compute.

```bash
python -m tasks.t3d_probe_converged_teacher --think_path $THINK --talk_path $TALK \
  --block_length 32 --top_k 10 --threshold 0.6 --gen_block 1 --limit 5
python -m tasks.t3d_position_profile --think_path $THINK --talk_path $TALK \
  --block_length 32 --top_k 10 --threshold 0.3 --gen_block 1 --limit 50
THINK=$THINK TALK=$TALK LIMIT=50 MODES="base_think think_then_talk" BLS="32 16 8" \
  bash scripts/sweep_block_size.sh
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
