# dFactory — DBet drafter training

Trains the **DBet drafter**: a small (~350M) trained Δh model bolted onto the **frozen 16B DMax-Math heavy**
(LLaDA2-MoE, block-diffusion). The drafter conditions on the heavy's selected hidden states (layers 1/10/19) +
last hidden + logits, and cheaply commits extra block tokens so the heavy runs fewer of its expensive forwards.
Inference lives in [`../dInfer`](../dInfer/README.md).

## Environment
Use the `dFactory` conda env. It needs a **prebuilt** flash-attn 2.x wheel matching your torch/CUDA (source builds
fail — no CUDA dev headers); VeOmni's model registry imports it transitively. torch 2.8 / cu12.8 / cp311 example:
```bash
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.x/flash_attn-2.8.x+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
python -c "import flash_attn.flash_attn_interface; print('ok')"
```

## Train (single GPU)
```bash
# 1) build the init checkpoint once: frozen heavy pieces + warm-started drafter layers
PYTHONPATH=$(pwd)/VeOmni:$(pwd):$PYTHONPATH python scripts/build_dbet_init.py \
    --heavy_path ../DMax-Math-16B-moe-merge --out_dir ./dbet_init \
    --draft_num_layers 5 --sel_layers 1,10,19
    # add --per_layer_denoise_fuse for the per-layer canvas-h_sel arch (build to ./dbet_init_pld instead)

# 2) train (config points model_path/config_path at the init dir)
PYTHONPATH=$(pwd)/VeOmni:$(pwd):$PYTHONPATH bash train.sh tasks/train_dbet.py configs/sft/dbet_bd_1gpu.yaml

# 3) watch it learn (metrics JSONL -> figures)
python scripts/plot_dbet_metrics.py --metrics ./dbet_outputs/dbet_metrics.jsonl
```
Checkpoints land in `<output_dir>/checkpoints/global_step_*/hf_ckpt` (small; saved every `save_steps`). **Judge the
drafter by GENERATION** (GSM8K accuracy + heavy-forward reduction via dInfer), **not** the teacher-forced val.

### Running a second variant without clobbering the first
Use a config with **isolated paths** — e.g. `configs/sft/dbet_bd_1gpu_pld.yaml` sets `config_path/model_path` →
`./dbet_init_pld` and `output_dir` → `./dbet_outputs_pld`, so its `checkpoints/` and `dbet_metrics.jsonl` never
overwrite the baseline run's.

## Layout
| path | what |
|---|---|
| `tasks/train_dbet.py` | VeOmni trainer entry point (the only training file you run) |
| `tasks/dbet_train_core.py` | `dbet_train_step` — the DBet loss (align-to-heavy CE + L1/TV, conf head, Δh) |
| `tasks/dbet_metrics.py` | train-metrics JSONL + held-out validation |
| `models/dbet/` | the drafter model (`modeling_dbet.py`, `configuration_dbet.py`) |
| `configs/sft/dbet_bd_1gpu*.yaml` | single-GPU configs (`_pld` = per-layer-denoise variant) |
| `configs/sft/dbet_bd.yaml` | multi-GPU (FSDP2 + meta init) config |
| `scripts/build_dbet_init.py` | build the init checkpoint (+ warm-start) |
| `scripts/plot_dbet_metrics.py` | metrics → figures |
| `attic/` | archived (`smoke_dbet.py` off-cluster smoke test, etc.) |

Legacy training entry points (heavy fine-tune / LLaDA2 block-diffusion) remain in `tasks/` as
`train_dmax_dbet_oput.py`, `train_llada2_bd*.py` — **not** part of the current drafter pipeline.
