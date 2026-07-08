# DBet — Self-Conditioned Δh Drafter  (`DBet` branch)

DBet speeds up the **frozen DMax-Math-16B heavy** (LLaDA2-MoE, block-diffusion) by training a small (~350M)
**drafter** that commits extra block tokens the heavy would otherwise decode itself — cutting the number of
expensive heavy forwards while staying iso-accurate. The heavy commits a confident left-to-right prefix in one
pass; the drafter proposes the remaining masked tokens (predicting Δh through the **frozen** LM head) plus a
trained confidence head that gates which draft commits to keep. **Only the drafter trains.**

- **Train the drafter** → [`dFactory/README.md`](./dFactory/README.md)
- **Evaluate / run inference** (heavy in SGLang + injected drafter) → [`dInfer/README.md`](./dInfer/README.md)

**Status:** iso-accurate (GSM8K 89%) under DMax's optimized cached + CUDA-graph SGLang decode, cutting heavy
forwards ~38% (80 → 49 per example) via a graph-safe feature tap that feeds the injected drafter.

Code: `dFactory/models/dbet/`, `dFactory/tasks/{train_dbet,dbet_train_core,dbet_metrics}.py` (training);
`dInfer/python/dinfer/decoding/generate_dbet_sglang_cached.py` + `model/modeling_llada2_moe_sglang_dbet.py`
(inference).

---

## Train the drafter

### 1. Build the init checkpoint (once)
The trainer loads `config_path == model_path == ./dbet_init`. Assemble it from the frozen heavy:
```bash
cd dFactory
PYTHONPATH=$(pwd)/VeOmni:$(pwd):$PYTHONPATH python scripts/build_dbet_init.py \
  --heavy_path ../DMax-Math-16B-moe-merge --out_dir ./dbet_init \
  --draft_num_layers 5 --sel_layers 1,10,19
  # add --per_layer_denoise_fuse (-> ./dbet_init_pld) for the per-layer canvas-h_sel arch
```

### 2. Train (single H200) with metrics + held-out validation
```bash
cd dFactory
PYTHONPATH=$(pwd)/VeOmni:$(pwd):$PYTHONPATH bash train.sh \
  tasks/train_dbet.py configs/sft/dbet_bd_1gpu.yaml
# quick figure run (~30-60 min): short warmup, capped steps, frequent eval
#   ... configs/sft/dbet_bd_1gpu.yaml --train.max_steps 3000 --train.lr_warmup_ratio 0.05 --train.eval_steps 250
```
`dbet_bd_1gpu.yaml` = 1-GPU (DDP, `init_device=cuda`). For ≥2 GPUs use `dbet_bd.yaml` (FSDP2 + meta). Set
`data.train_path` to your OPUT data. Run a second variant without clobbering the first via an isolated-paths
config (e.g. `dbet_bd_1gpu_pld.yaml` → `dbet_init_pld` / `dbet_outputs_pld`).

### 3. Metrics & figures
Everything logs to **`<output_dir>/dbet_metrics.jsonl`** (`split` = `train`|`val`); the held-out eval re-uses the
exact training forward under `no_grad`. **Judge the drafter by GENERATION** (GSM8K accuracy + heavy-forward
reduction via dInfer), not the teacher-forced val.

| metric | split | meaning |
|---|---|---|
| `loss`, `tok`, `conf` | train | total / token-CE / confidence-BCE loss |
| `acc` | train/val | drafter accuracy on remaining-masked positions (val = vs golden) |
| `auc` | val | confidence-head ROC-AUC (predicting drafter-correctness); ship gate > 0.7 |
| `acc_sig{σ}`, `auc_sig{σ}`, `acc_by_pos` | val | accuracy/AUC per swept mask ratio σ, and vs distance-into-block |

```bash
python scripts/plot_dbet_metrics.py --metrics ./dbet_outputs/dbet_metrics.jsonl   # -> ./dbet_outputs/figures/
```
Eval knobs (yaml or `--train.<name>`): `log_steps`, `eval_steps`, `eval_holdout_size`, `eval_sigmas`,
`eval_at_start`, `skip_nonfinite_steps`. See `dFactory/README.md` for the full layout.

## Evaluate
See [`dInfer/README.md`](./dInfer/README.md). In short — heavy in SGLang + the drafter injected into DMax's
cached+graph block-diffusion decode:
```bash
cd dInfer
python evaluations/eval_dbet_sglang_cached_gsm8k.py \
  --heavy_path ../DMax-Math-16B --drafter_path ../dFactory/dbet_outputs/checkpoints/<step>/hf_ckpt \
  --out_path preds.jsonl --limit 100 --cuda_graph --compile_draft
python evaluations/val_gsm8k.py --pred-path preds.jsonl --limit 100
#   --no_draft = pure heavy-only baseline in the same decode (for the fwd/ex comparison)
```

## Layout
```
├── dFactory/        drafter training (models/dbet, tasks/train_dbet, configs/sft/dbet_bd*)  -> dFactory/README.md
├── dInfer/          inference + eval (decoding/generate_dbet*, model/*_sglang_dbet)          -> dInfer/README.md
└── DMax-Math-16B*   the frozen heavy checkpoints (per-expert + merged)
```

## Acknowledgements
Builds on **DMax: Aggressive Parallel Decoding for dLLMs** (Chen, Fang, Ma, Yu, Wang; NUS, 2026),
<https://github.com/czg1225/DMax> (Apache-2.0). We reuse and adapt its `dFactory/` training pipeline, OPUT data
processing, block-diffusion training/decoding, and the `LLaDA2Moe*` modeling code. Files derived from DMax carry
the original copyright in their headers.

## License
Apache-2.0.
