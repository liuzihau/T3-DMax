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

# 4. Launch milestone-1 training
PYTHONPATH=$(pwd)/VeOmni:$PYTHONPATH \
sh train.sh \
  tasks/train_t3_dmax_bd_oput.py \
  configs/sft/t3_llada2_mini_bd_oput.yaml
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
