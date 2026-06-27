# DBet — branch plan & high-level inventory

**Branch:** `DBet` (off T3-DMax `main`). Builds the **Self-Conditioned Δh Drafter** on T3-DMax's existing
drafter-training stack, fixing the design choices that made T3-DMax fail. Full design:
`../../fork_bounded_surrogate/draft_model_design.md`. Why build here:
`../../fork_bounded_surrogate/draft_model_design.md` Part II (T3-DMax already has ~80% of the stack).

Heavy/corrector = **DMax-Math-16B, frozen** (we never retrain it). Only the drafter trains.

---

## 1. Original DMax code (the clean base — now recollected onto this branch)

T3-DMax kept DMax's model + dataset + dInfer, but had **dropped** DMax's training recipe. Restored here for
reference (the OPUT/SFT scaffold our drafter trainer will mirror):

| path | what it is |
|---|---|
| `dFactory/models/llada2_moe/` | the LLaDA-2.0-MoE backbone (= DMax/heavy). NOTE: T3 modified `modeling_llada2_moe.py` (+8 lines vs DMax). |
| `dFactory/tasks/train_llada2_bd.py` | **[recollected]** DMax block-diffusion SFT trainer |
| `dFactory/tasks/train_llada2_bd_oput.py` | **[recollected]** DMax **OPUT** trainer (the self-correction recipe) |
| `dFactory/tasks/train_llada2_bd_with_dparallel.py` | **[recollected]** DMax dparallel trainer |
| `dFactory/configs/sft/llada2_mini_bd_oput.yaml` | **[recollected]** OPUT config |
| `dFactory/configs/sft/llada2_mini_bd_sft.yaml`, `llada2_flash_bd_sft.yaml` | **[recollected]** SFT configs |
| `dFactory/configs/model_configs/llada2_mini`, `llada2_flash` | DMax model configs (already present) |
| `dFactory/tasks/dataset/{dataset.py,data_transform.py}` | data pipeline (already present) |
| `dInfer/` | DMax inference framework (already present) |

*(Pristine DMax for diffing lives at `../../related_research/decode_algorithm/DMax/DMax/`.)*

## 2. Our model structure + config (what we extend)

| path | what it is | role in our design |
|---|---|---|
| `dFactory/models/think_talk_llada2/modeling_think_talk_llada2.py` | T3's talk model: `AnchorFuser`, `GatedResidualConditioning`, `TalkCrossAttention`, `TalkDecoderLayer` (dense, LLaDA-2.0 primitives), `use_anchor_delta_head` (zero-init Δh + frozen lm_head) | **≈80% of our drafter** — extend this |
| `dFactory/models/think_talk_llada2/configuration_think_talk_llada2.py` | talk config (delta-head flag, layer selection, …) | our model config (pluggable layers) |
| `dFactory/configs/model_configs/think_talk_llada2_mini*` | talk model_configs (several variants incl. `frozen_think`, `xattn_talk4`) | starting points |
| `../../probe_runner/draft_model.py` | **our standalone Part-I prototype** (validated 2026-06-27: wiring + confidence head, toy conf_auc 0.57→0.996) | architecture reference; the *real* model = extended `think_talk_llada2` |

## 3. T3-DMax training stack to REUSE

| path | what it is |
|---|---|
| `dFactory/tasks/train_t3_topk_talk.py` | VeOmni training loop: frozen `think`(DMax) via `think_path`, anchor extraction, talk training |
| `dFactory/configs/sft/t3_topk_talk_stage1_coldstart.yaml` | **Stage-1 cold start** (off-policy) — our Part-II Stage-1 |
| `dFactory/configs/sft/t3_topk_talk_stage2_onpolicy.yaml` | **Stage-2 on-policy** — our Part-II Stage-2 (DAgger) |
| `dFactory/tasks/t3d_topk_soft_embed.py` | top-K soft-embed task (we replace top-K blur with full `softmax·W_E`) |
| `dFactory/tasks/t3d_topk_talk.py`, `t3d_topk_eval_gsm8k.py` | talk decode + GSM8K eval scaffolding |
| `VeOmni/` | the distributed training framework everything runs on |

## 4. Keep / Change / Add (our corrected design vs T3-DMax)

| component | T3-DMax has | DBet action |
|---|---|---|
| frozen think (DMax) loading, anchor extraction | ✅ | **KEEP** |
| two-stage cold-start → on-policy configs | ✅ | **KEEP** (re-point to our task) |
| Δh + frozen lm_head (`use_anchor_delta_head`) | ✅ | **KEEP** |
| AnchorFuser / GatedResidualConditioning / TalkCrossAttention | ✅ | **KEEP** as the conditioning base |
| config-driven thin trainable layers (`t3_train_layers`) | ✅ | **KEEP** (= pluggable layers) |
| soft-embed of heavy logits | top-K **blur** (collapsed) | **CHANGE → full `softmax(ℓ/τ)·W_E`** (DiffusionGemma) |
| role of drafter | **committer** (under-commit collapse) | **CHANGE → proposer; heavy verifies (agreement-accept)** |
| context injection | cross-attn / anchor | **ADD DFlash per-layer KV injection + per-layer D3 fuse** |
| confidence / abstention | ✗ none | **ADD trained confidence head** (asymmetric loss — the B1 fix) |
| heavy training | (T3 retrained talk) | **KEEP heavy FROZEN** (no OPUT retrain) |

## 5. Next steps on DBet
1. Read-and-map `modeling_think_talk_llada2.py` in detail → exact edit list (TalkCrossAttention → per-layer KV;
   add full soft-embed; add confidence head) — Part II concretization.
2. New model = extended `think_talk_llada2`; new task = a `train_dbet.py` adapted from `train_t3_topk_talk.py`;
   new configs = Stage-1/Stage-2 adapted from the t3 ones.
3. Phase-A one-shot inference loop (extend `exp_b2_selfspec.py` or a dInfer mode) → iso-accuracy gate.

> Branch hygiene: changes uncommitted. Pristine DMax is external (`related_research/.../DMax`); this branch only
> *adds* the recollected files + this plan, nothing in T3's own code was overwritten.
