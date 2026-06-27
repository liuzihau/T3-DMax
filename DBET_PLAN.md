# DBet — branch plan & high-level inventory

**Branch:** `DBet` (off T3-DMax `main`). Builds the **Self-Conditioned Δh Drafter** on T3-DMax's existing
drafter-training stack, fixing the design choices that made T3-DMax fail. Full design:
`../../fork_bounded_surrogate/draft_model_design.md`. Why build here:
`../../fork_bounded_surrogate/draft_model_design.md` Part II (T3-DMax already has ~80% of the stack).

Heavy/corrector = **DMax-Math-16B, frozen** (we never retrain it). Only the drafter trains.

**Naming + branch hygiene (2026-06-28):** our new model/configs use the **`dbet_llada2`** namespace (built
fresh — we do NOT edit `think_talk_llada2`). All `think_talk_llada2` / `t3d_*` / `t3_topk_*` files were **removed
from this branch** to keep it focused (we haven't done training-method yet). They remain on `main` for reference
— restore any with `git checkout main -- <path>`, or read via `git show main:<path>`.

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

We build a **NEW `dbet_llada2`** model + configs (fresh namespace), *referencing* think_talk's good parts on `main`
(not editing them). Targets to create on this branch:

| path (to CREATE) | what it is |
|---|---|
| `dFactory/models/dbet_llada2/modeling_dbet_llada2.py` | our drafter: per-layer KV injection (DFlash) + per-layer D3 fuse + full soft-embed + Δh + **confidence head** |
| `dFactory/models/dbet_llada2/configuration_dbet_llada2.py` | our config: `sel_layers`, pluggable `layer_type`, per-layer-fuse flag, soft-embed τ, conf-head, frozen-head flags |
| `dFactory/configs/model_configs/dbet_llada2_mini` | model_config for the mini drafter |

Reference material (NOT in this branch — read from `main`):
| ref (on `main`) | why |
|---|---|
| `git show main:dFactory/models/think_talk_llada2/modeling_think_talk_llada2.py` | `AnchorFuser`, `GatedResidualConditioning`, `TalkCrossAttention`, `TalkDecoderLayer`, `use_anchor_delta_head` (zero-init Δh + frozen lm_head) — copy the good parts, fix the failed ones |
| `../../probe_runner/draft_model.py` | our standalone Part-I prototype (validated: wiring + conf head, toy conf_auc 0.57→0.996) — the architecture spec in code |

## 3. Training stack — DEFERRED (training method not started; refs on `main`)

We have NOT designed the training method on this branch yet. When we do, adapt these from `main` into `dbet_*`:
| ref (on `main`) | what it is |
|---|---|
| `git show main:dFactory/tasks/train_t3_topk_talk.py` | VeOmni loop: frozen `think`(DMax) via `think_path`, anchor extraction → adapt to `train_dbet.py` |
| `git show main:dFactory/configs/sft/t3_topk_talk_stage1_coldstart.yaml` / `…stage2_onpolicy.yaml` | two-stage schedule → adapt to `dbet_*` configs |
| `dFactory/tasks/train_llada2_bd_oput.py` (**recollected, in-branch**) | DMax's own OPUT/SFT trainer — reference recipe |
| `VeOmni/` (in-branch) | the distributed training framework |

## 4. Port / Change / Add — building `dbet_llada2` (reference = think_talk on `main`)

"PORT" = re-implement the good part fresh in `dbet_llada2` (reference think_talk on `main`), not edit in place.

| component | on `main` (think_talk) | DBet action (in `dbet_llada2`) |
|---|---|---|
| frozen think (DMax) loading, anchor extraction | ✅ | **PORT** |
| two-stage cold-start → on-policy configs | ✅ | **PORT** (deferred — training method §3) |
| Δh + frozen lm_head (`use_anchor_delta_head`) | ✅ | **PORT** |
| AnchorFuser / GatedResidualConditioning / TalkCrossAttention | ✅ | **PORT** as the conditioning base |
| config-driven thin trainable layers (`t3_train_layers`) | ✅ | **PORT** (= pluggable `layer_type`) |
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
