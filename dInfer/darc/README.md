# DARC — Diffusion Auto-Regressive Chain

A small autoregressive **refinement head** grafted into a frozen DMax (LLaDA2-MoE) diffusion LLM. It re-reads
each block **left-to-right**, using the model's own top-k belief as committed context, then fuses the cleaned
sequence back into the final transformer layer. Goal: recover the accuracy that parallel diffusion decoding
loses at later block positions — cheaply, with a frozen backbone.

Builds on the T3-DMax code (branched from `DBet`). The dead DBet experiment is left untouched in history.

## Why (the probe result that motivates this)

`../evaluations/probe_layer_readout.py` (results in `../runs/`) showed, for the frozen DMax-Math-16B model:

- **Single-forward ranking collapses left→right in a block.** Last-layer recall@1 for masked positions:
  pos0 ≈ 0.94 → pos30 ≈ 0.16.
- **But the gold token stays in a tiny candidate set:** recall@500 ≈ 0.99 everywhere; nucleus(0.5)∩top-500
  averages ~1–2 tokens wide.
- **The collapse is error-compounding, not intrinsic.** Conditioned on a *clean* left prefix (all earlier
  in-block positions correct), late-block recall@1 jumps **0.23 → 0.76** (recall@5 ≈ 0.98).
- Readouts only concentrate at the **last ~2 layers** (L18–L19), so the head taps **after layer 18**.

DARC is the mechanism that supplies that clean left context — autoregressively.

## Architecture

Tap the hidden states at the **output of layer 18** (`h_0..h_{n-1}` per block). Insert three trainable
modules before layer 19:

1. **New attention block** (cross-attention-like): query from the current position's `h_i`; keys/values from
   the committed **soft-embeds** `s_{j<i}`. Reuses the base RoPE/LN conventions. No KV cache — Q/K/V are
   recomputed over the whole prefix each step (identical result, and it *is* the parallel-trainable form).
2. **New MLP** (SwiGLU).
3. **Fuse**: maps the soft-embed sequence back into residual-stream space (keep a residual to `h_i`) → input
   to layer 19.

**Soft-embed** = top-k (start k=5) softmax-weighted sum over the **frozen base input embedding** (NOT a new
trainable table — the trainable W_kv/fuse carry all needed adaptation; a full table is 322M sparse params).
**LM head frozen.** Trainable ≈ **50M** (attn ~10.5M + MLP ~31.5M + fuse ~8.4M); optional LoRA on the last
1–2 layers (~1–2M) and/or an embed *adapter* (D×D ~4M).

### Forward (per block, left-to-right)
- **pos 0 — frozen seed:** `s_0 = SoftEmbed(LMhead(h_0))`. No trainable params, no loss (pos0 recall@1≈0.94).
- **pos i ≥ 1:** `h_i` attends to `s_{<i}` → MLP → LMhead → CE(gold_i) → SoftEmbed → detach → `s_i`.
- **fuse** `s_0..s_{n-1}` with `h` → layer 19 → … → final norm → LM head.

Inference is sequential (`s_i` needs `s_{i-1}`); training is parallel (see below).

## Training (self-conditioned, stop-gradient)

Teacher-forcing the soft-embeds is impossible (there is no "correct" soft-embed label), so we run the real AR
chain and control gradients with **detach**. Two losses:

- **Loss 1 (local, trains attn+MLP):** per-position CE against gold, with every context soft-embed
  **detached**. Because contexts are detached, after a sequential produce-the-soft-embeds pass, all positions'
  logits+CE compute in **one parallel causal-masked forward** (pack many blocks side by side). Detach only
  changes the backward pass — forward values match inference exactly, so **no exposure bias**.
- **Loss 2 (global, trains fuse):** fuse the soft-embed sequence, run through L19→…→final norm→LM head, CE
  against gold. (First trial: soft-embeds detached → Loss 2 trains the fuse only.)

**Target = self-gold** (the model's own decode at threshold 0.5), not dataset ground truth — DARC is a
shortcut to the model's *own* output, not a new capability.

**First-trial simplification:** train on a **single forward, whole block masked** (the forward-1 collapse
case). Success metric: **accuracy lift per unit added sequential latency** after one forward — not accuracy
alone.

### Train/inference consistency notes
- Main mismatch to watch: all-masked training vs multi-forward inference (some positions revealed as *hard*
  tokens later). Fine for the first trial **if evaluated one-forward all-masked**; for the full loop, mix
  hard (revealed) + soft (masked) context embeds in training.
- Keep **k** and the hard-vs-soft propagation rule **identical** in train and inference.
- Non-stationary context early on (untrained head → garbage soft-embeds): optional warm-up with a clean
  hard-embed prefix, annealed to self-produced.

## Data

`../evaluations/collect_gold_data.py` — DMax self-gold from `nvidia/Nemotron-Post-Training-Dataset-v2`
(config `SFT`, split `math`, 239,467 prompts), `gen_length=512`. Deterministic global permutation fixed by
`--seed`; collect rank range `[--start,--end)` so **5k→50k extends without recomputing** the first 5k. Store
`{prompt_ids, gold_ids, eos_cut, ...}` per line (self-contained). Sharded + resumable.

- Smoke: 150 (already have from the probe). First trial: **3k–5k** examples (~0.6–1M targets; ×4–8 via
  mask-pattern augmentation). Scale to 15k–30k only for a general/multi-domain head.

## Layout
```
darc/
  README.md          # this charter
  model/ar_head.py   # DARC head module (attn + MLP + fuse, soft-embed, causal-masked forward)   [stub]
  train/             # two-phase training (produce soft-embeds → detached Loss 1 + fused Loss 2)  [todo]
  data/              # loader over collect_gold_data.py shards → block-structured targets          [todo]
  configs/           # trial configs (k, ranks, LoRA rank, losses)                                 [todo]
```
Data collection currently lives at `../evaluations/collect_gold_data.py` (imports the probe's byte-for-byte
DMax decode); it will be wrapped by `darc/data/` for training.

## Status
- [x] Prune-validity probe → go signal (last-2-layer readout, clean-prefix recovery 0.23→0.76).
- [x] Data-collection pipeline (seeded, shardable, extensible).
- [ ] DARC head module.
- [ ] Two-phase training loop.
- [ ] First trial: one-forward all-masked, measure accuracy lift vs added latency.
