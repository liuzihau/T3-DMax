# Heavy fine-tune — degeneration ablation (SCHEDULED, needs 2 GPUs)

**Status:** parked, waiting on the 2-GPU server. Run when it frees up.

## The problem
The 40k heavy fine-tune (routes B+A+C, lr 2e-6) produced **great teacher-forced val** metrics but its
**heavy-only free generation collapses** into repetition (e.g. `sprint sprint … meters meters`). The val is
teacher-forced (golden revealed prefix + golden clean stream) so it **cannot see** the collapse.

Val step0 → 40k: `loss_C 0.707→0.442 (−37%)`, `acc_corr2 0.876→0.887`, `acc_heavy3 0.893→0.893` — all "nice",
generation dead. Note the **prior** run barely trained (flat loss, bad LR schedule) so its generation stayed
~DMax; THIS run actually learned → the heavy moved → collapse. So the real finding is *"when the heavy actually
moves on our objective, generation breaks."*

## What we already ruled out (static analysis)
- **Loss gate is IDENTICAL to DMax OPUT.** Ours: target `clean_ids`, gate `w=(noisy_ids==mask_id)` → masked-only.
  DMax (`train_llada2_bd_oput.py:490`): `loss/(labels!=-100)`, and `data_transform_dbet.py:117`
  `labels[~loss_mask]=-100  # loss only at MASK positions`. Both masked-only, LLaDA objective. **Not the bug.**
- The dual-stream golden "clean" crutch is **also in DMax** (its rollout forward keeps the golden clean half), so
  it doesn't explain why *ours* breaks.

## The concrete deltas of our objective vs DMax OPUT (ranked suspects)
1. **Route A (draft-contaminated rollout)** — the one thing DMax does NOT have.
2. **Soft-embed rollout (Route C, top-1 soft) vs DMax's HARD-argmax rollout** (`train_llada2_bd_oput.py:462`).
   Caveat: DMax trains HARD but infers SOFT; we train SOFT + infer SOFT → we're *more* consistent, so this is a
   weaker suspect (tier-2).
3. **All routes every step (summed), ignoring the on/off `flag`** vs DMax's stochastic mask-XOR-rollout.
4. **top-12 partial FT + left-to-right reveal + noise_range 0.75–1.0** (0–25% reveal vs DMax's ~25% random).

## The ablations (configs already committed)
Identical to the broken 40k run except the loss weights + a distinct output_dir, so the ONLY variable is the
ablated route. **Judge by GENERATION (`analyze_gen_lengths` degen% + eyeballing answers), NOT the val.**

### Run 1 — `ablBC` (B+C, Route A OFF) — highest-info, run first
```bash
cd /scratch/rds-fei-goodspeed-rw/nick/T3-DMax/dFactory && git pull
PYTHONPATH=$(pwd)/VeOmni:$(pwd):$PYTHONPATH bash train.sh tasks/train_dmax_dbet_oput.py configs/sft/dmax_dbet_oput_ablBC.yaml
# mid-run (step 20000 ckpt lands via save_steps): extract + test heavy-only generation
python scripts/extract_heavy_from_ft.py --ft_ckpt ./dmax_ft_bc/checkpoints/global_step_20000/hf_ckpt \
    --dmax_ref ../DMax-Math-16B-moe-merge --out_dir ./heavy_bc_20k
# in dInfer: point heavy= at ../dFactory/heavy_bc_20k, run heavy-only sweep configs, then:
python evaluations/analyze_gen_lengths.py --dir <sweep_out>
```

### Run 2 — `ablB` (B only, pure mask-denoise) — run only if ablBC still breaks
```bash
PYTHONPATH=$(pwd)/VeOmni:$(pwd):$PYTHONPATH bash train.sh tasks/train_dmax_dbet_oput.py configs/sft/dmax_dbet_oput_ablB.yaml
# extract ./dmax_ft_b/checkpoints/global_step_20000/hf_ckpt -> sweep -> analyze_gen_lengths
```

## Decision tree
- **ablBC generation OK** (degen ~DMax) → **Route A (draft) is the poison** → drop/redesign A. Done.
- **ablBC breaks** → run ablB:
  - **ablB OK** → **rollout routes (soft C) are the issue** → tier-2: add `heavy_rollout_hard` (hard argmax
    like DMax) and retest.
  - **ablB breaks** → the **top-12 partial FT / reveal / lr** itself destabilizes the heavy (deepest finding).

## Tier-2 / adjacent checks (do after the above localizes it)
- **hard-argmax rollout** flag for the heavy commit in A/C (matches DMax's recipe).
- **Align train soft-embed to inference:** training uses `heavy_soft_tau=1.0`, but DMax config has
  `soft_embed_temp=0.8`. If the sweep decodes at τ=0.8 we trained the heavy for the wrong soft-embed. Check what
  `heavy_tau` the sweep passes and align `heavy_soft_tau` to it.
- **Always** add a free-generation eval (decode ~10–20 GSM8K during training) — the val alone is blind to this.

Anchor every sweep with the **original DMax heavy** row and the frozen original drafter.
