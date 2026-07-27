# DARC — Diffusion Auto-Regressive Chain

A small autoregressive **refinement head** grafted into a frozen DMax (LLaDA2-MoE) diffusion LLM. It re-reads
each block **left-to-right**, using the model's own top-k belief as committed context, then fuses the cleaned
sequence back into the final transformer layer. Goal: recover the accuracy that parallel diffusion decoding
loses at later block positions — cheaply, with a frozen backbone.

Builds on the T3-DMax code (branch `DARC`, cut from the dead `DBet`). This directory is the project **hub**;
code is distributed across the repo's standard `dFactory` (train + data) / `dInfer` (inference) split — see
Layout below.

## Why (the probe result that motivates this)

`dInfer/evaluations/probe_layer_readout.py` (results in `dInfer/runs/`) showed, for the frozen
DMax-Math-16B model:

- **Single-forward ranking collapses left→right in a block.** Last-layer recall@1 (masked): pos0 ≈ 0.94 →
  pos30 ≈ 0.16.
- **But the gold token stays in a tiny candidate set:** recall@500 ≈ 0.99 everywhere; nucleus(0.5)∩top-500
  averages ~1–2 tokens.
- **The collapse is error-compounding, not intrinsic.** Conditioned on a *clean* left prefix, late-block
  recall@1 jumps **0.23 → 0.76** (recall@5 ≈ 0.98).
- Readouts only concentrate at the **last ~2 layers** (L18–L19) → the head taps **after layer 18**.

DARC is the mechanism that supplies that clean left context — autoregressively.

## Architecture

Tap layer 18's output (`h_0..h_{n-1}` per block). Insert three trainable modules before layer 19:

1. **New attention** (cross-attention-like): query from `h_i`; keys/values from committed **soft-embeds**
   `s_{j<i}`. Reuses base RoPE/LN. No KV cache — Q/K/V recomputed over the prefix each step (identical result,
   and it *is* the parallel-trainable form).
2. **New MLP** (SwiGLU).
3. **Fuse**: soft-embed sequence → residual space (keep a residual to `h_i`) → input to layer 19.

**Soft-embed** = top-k (start k=5) softmax-weighted sum over the **frozen base input embedding** (NOT a new
trainable table). **LM head frozen.** Trainable ≈ **50M** (attn ~10.5M + MLP ~31.5M + fuse ~8.4M); optional
LoRA on the last 1–2 layers (~1–2M) and/or an embed *adapter* (D×D ~4M).

### Forward (per block, left-to-right)
- **pos 0 — frozen seed:** `s_0 = SoftEmbed(LMhead(h_0))`. No trainable params, no loss (pos0 recall@1≈0.94).
- **pos i ≥ 1:** `h_i` attends to `s_{<i}` → MLP → LMhead → CE(gold_i) → SoftEmbed → detach → `s_i`.
- **fuse** `s_0..s_{n-1}` with `h` → layer 19 → … → final norm → LM head.

Inference is sequential (`s_i` needs `s_{i-1}`); training is parallel (below).

## Training (self-conditioned, stop-gradient)

Teacher-forcing the soft-embeds is impossible (no "correct" soft-embed label), so we run the real AR chain and
control gradients with **detach**. Two losses:

- **Loss 1 (local, trains attn+MLP):** per-position CE vs gold with every context soft-embed **detached**.
  After a sequential produce-the-soft-embeds pass, all positions' logits+CE compute in **one parallel
  causal-masked forward** (pack many blocks side by side). Detach only changes the backward pass — forward
  values match inference exactly, so **no exposure bias**.
- **Loss 2 (global, trains fuse):** fuse → L19→…→final norm→LM head, CE vs gold. (First trial: soft-embeds
  detached → Loss 2 trains the fuse only.)

**Target = self-gold** (the model's own decode at threshold 0.5), not dataset ground truth.

**First-trial simplification:** single forward, whole block masked. Metric: **accuracy lift per unit added
sequential latency** after one forward — not accuracy alone.

### Train/inference consistency notes
- Main mismatch: all-masked training vs multi-forward inference (later forwards reveal *hard* tokens). Fine
  for the first trial **if evaluated one-forward all-masked**; for the full loop, mix hard+soft context embeds.
- Keep **k** and the hard-vs-soft propagation rule **identical** in train and inference.
- Non-stationary context early on: optional warm-up with a clean hard-embed prefix, annealed to self-produced.

## Data

`dFactory/scripts/collect_gold_data.py` — DMax self-gold from `nvidia/Nemotron-Post-Training-Dataset-v2`
(config `default`; `math` split-or-category, ~239k prompts), `gen_length=512`. Reuses dInfer's byte-for-byte
DMax decode. Deterministic global permutation fixed by `--seed`; collect rank range `[--start,--end)` so
**5k→50k extends without recomputing** the first 5k. Sharded + resumable. Stores `{prompt_ids, gold_ids,
eos_cut, ...}` per line (self-contained). Default output → `dFactory/darc_gold/`.

- Smoke: 150 (probe). First trial: **3k–5k** examples (~0.6–1M targets; ×4–8 via mask-pattern augmentation).
  Scale to 15k–30k only for a general/multi-domain head.

## Layout (follows the repo's dFactory=train+data / dInfer=inference split)
```
dFactory/
  models/darc/               # THIS dir — head module (mirrors models/dbet/)
    README.md                #   project hub (this file)
    configuration_darc.py    #   head config (DarcConfig dataclass)                              [done]
    modeling_darc.py         #   head: AR attn + MLP + fuse, soft-embed, Loss-1 forward          [done: Loss-1]
  scripts/collect_gold_data.py   # self-gold data collection (mirrors build_*_dataset.py)
  tasks/                     # [todo] train_darc.py + darc_train_core.py (mirror train_dbet.py)
  tasks/dataset/data_transform_darc.py   # gold shards -> {input_ids,noisy,labels,attn}  [done]
  configs/sft/               # [todo] darc_*.yaml trial configs
dInfer/
  evaluations/probe_layer_readout.py   # the prune-validity probe (analysis; stays here)
  <later>                    # head inference integration into the DMax decode loop
```

## Status
- [x] Prune-validity probe → go signal (clean-prefix recovery 0.23→0.76).
- [x] Data-collection pipeline (seeded, shardable, extensible), now in dFactory/scripts.
- [x] Dataset transform (`tasks/dataset/data_transform_darc.py`) — gold shards -> training tensors
      (all-masked first-trial mode + ltr_reveal/random for later; borrows DMax masking; self-tested).
- [x] DARC head module — Loss-1 (`modeling_darc.py` + `configuration_darc.py`): self-conditioned
      stop-gradient AR, block-strict-causal, frozen seed/hard-embed handling; model-free self-test.
- [ ] Base-model integration: import LLaDA2Moe pieces, cos/sin from LLaDA2MoeRotaryEmbedding, tap via
      `heavy_forward(output_hidden_states=True)`, resolve `tap_layer` vs the probe.
- [ ] Loss-2 (fuse -> L19+ replay) + block-parallel produce optimization.
- [x] Integration smoke (`scripts/smoke_darc_integration.py`): tap-18 verified; reveal-prior reproduces the
      probe (avg 0.44, p0=0.95→p30=0.15) vs all-masked 0.12; head runs on real bf16, backbone frozen.
- [x] Single-GPU first-trial trainer (`tasks/train_darc.py`): frozen backbone + Loss-1 head, reveal-prior
      data, deterministic rank-based train/val split, val acc1 vs probe baselines, head checkpoints.
- [ ] First-trial run on 10k gold; watch val acc1 climb from ~0.23 (uncond) toward ~0.76 (clean-prefix).
- [ ] Loss-2 (fuse->L19+) + block-parallel perf + (later) multi-GPU FSDP via the DBet harness.
