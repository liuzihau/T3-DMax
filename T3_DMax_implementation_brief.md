# T3-DMax Adaptation Implementation Brief

## 0. Purpose

This document is an implementation brief for a coding agent. The goal is to build a new experimental codebase that combines:

1. **DMax training infrastructure**: block-diffusion SFT, On-Policy Uniform Training (OPUT), public DMax trajectory data, full fine-tuning, distributed training/checkpointing.
2. **Think-Then-Talk (T3) architecture**: a heavy diffusion-LM backbone produces a single per-block "thinking anchor" (last-layer hidden state); a lightweight talk model iteratively decodes the masked block conditioned on that anchor.

The target experiment is:

> Keep DMax's data, hyperparameters, block-diffusion objective, and OPUT schedule as close as possible. Replace only the model architecture with Think-Then-Talk, and adapt the OPUT rollout to T3's compute pattern (think runs once per block; talk performs all iterations).

### 0.1 Primary hypothesis (H1 — architecture viability)

> Can a (heavy think backbone + lightweight talk model) split match DMax-comparable answer quality on LLaDA-2.0-mini, while reducing per-step inference compute by amortising the heavy backbone across all iterations of a block?

The baseline is **LLaDA-2.0-mini under DMax OPUT** (~92% GSM8K, TPF 5.48). We are not chasing the 7pp gap that the prior LLaDA-8B-based Think-Then-Talk runs hit — that was a separate quality blocker against a different backbone, and is parked.

### 0.2 Decision log (round 1)

Locked-in choices that differ from a literal DMax adaptation:

| # | Decision | Choice | Rationale |
|---|---|---|---|
| 1 | Anchor fuser | **Last-layer hidden state only** | EAGLE/DFlash 3-layer concat was speculative-decoding heritage; not the right inductive bias for a denoising anchor. Keep the code path scalable to multi-layer fuse for later trials. |
| 2 | OPUT rollout | **Talk-only with cached anchor** | Think has no LM head — only talk can produce tokens. No-grad rollout reuses the think anchor computed on the masked input; grad forward runs talk on the predicted input. Saves ~50% wall-clock on `flag=True` samples vs. recomputing think. |
| 3 | Block size | **K = 32 from the start** | DMax shows K=32 is viable at first decode; no reason to start smaller. |
| 4 | Talk depth | **2 layers** (was 4 in v1) | Matches current T3 codebase; one architectural variable changed at a time. Multi-layer talk is ablation A4. |
| 5 | Loss | **Uniform CE** | Step-weighted loss (current T3's `step_agg`, `step0_boost`) is parked. Re-introduce only if the strict run underperforms. |
| 6 | SPD scope | **Soft embeddings into talk only** | Think only ever sees fresh all-mask blocks at inference; talk is the only model that benefits from hybrid-embedding inputs. |
| 7 | Quality target | **DMax-comparable accuracy at lower per-step inference compute** | The compute-savings claim is the point of the experiment. Matching DMax accuracy exactly is unnecessary; coming within ~1–2pp is acceptable if TPF or per-step cost wins. |

Parked open questions:

- **Q-think-prune**: whether to prune LLaDA-2.0-mini's last layer and transplant it as the talk model's initial weights (current Think-Then-Talk does this). Default for milestone 1: **think = full LLaDA-2.0-mini, talk = fresh-init 2-layer transformer**. Promoted to ablation A1.5.

---

## 1. Recommended repository strategy

### 1.1 Do not start from an empty repo

The DMax training stack depends on its `dFactory` layout, VeOmni integration, model registry, config paths, training scripts, checkpoint conversion, and MoE merged-expert format. Recreating this from scratch would waste time and create unnecessary bugs.

### 1.2 Recommended option: create a new repo based on DMax

Recommended workflow:

```bash
git clone --recursive https://github.com/czg1225/DMax.git T3-DMax
cd T3-DMax

# Remove original remote or rename it
git remote rename origin dmax-upstream

# Create a new GitHub repo, e.g. liuzihau/T3-DMax, then:
git remote add origin git@github.com:liuzihau/T3-DMax.git

git checkout -b t3-dmax-main
```

Then add T3-specific code into the existing DMax/dFactory structure.

Why this is better:

- DMax already has the correct training entry points.
- DMax already has the correct data processing scripts.
- DMax already supports the LLaDA2.0-mini MoE model layout.
- DMax already supports full fine-tuning and HF checkpoint export.
- The coding agent only needs to implement new model classes and a modified training script.

### 1.3 Alternative option: fork DMax directly

This is also acceptable:

```bash
# Fork czg1225/DMax on GitHub to liuzihau/DMax-T3
git clone --recursive git@github.com:liuzihau/DMax-T3.git
cd DMax-T3
git checkout -b t3-dmax-main
```

This is the fastest route if the project is mostly an adaptation of DMax.

### 1.4 Where to place borrowed T3 code

Do not copy the whole Think-Then-Talk repo blindly. Instead, copy only the useful components:

- fuser / anchor projection logic
- lightweight talk model design
- iterative denoising utilities
- evaluation utilities if useful

Recommended destination:

```text
dFactory/models/think_talk_llada2/
    __init__.py
    configuration_think_talk_llada2.py
    modeling_think_talk_llada2.py
    fusers.py
    talk_layers.py
```

Add new training task:

```text
dFactory/tasks/train_t3_dmax_bd_oput.py
```

Add new config:

```text
dFactory/configs/sft/t3_llada2_mini_bd_oput.yaml
```

Add new inference/eval later:

```text
t3_infer/
    generate_think_talk.py
    eval_gsm8k.py
```

---

## 2. Licence and attribution requirements

DMax is released under Apache-2.0. Preserve its `LICENSE` file and any copyright notices.

Since this new repo derives heavily from DMax, add a section in the new README:

```markdown
## Acknowledgements

This repository builds on DMax: Aggressive Parallel Decoding for dLLMs
(https://github.com/czg1225/DMax), released under Apache-2.0.
We reuse and adapt its dFactory training pipeline, OPUT data processing,
and block-diffusion training scripts.

This repository also reuses components from Think-Then-Talk
(https://github.com/liuzihau/Think-Then-Talk).
```

If Think-Then-Talk does not currently have a licence file, add one before copying code into a public combined repo. Since the repo belongs to the project owner, this is mainly to make reuse clear for future collaborators.

---

## 3. Baseline facts to preserve from DMax

The first T3-DMax experiment should preserve these DMax settings as much as possible:

```yaml
block_diffusion_mode: true
block_size: 32
same_token_labels: true
noise_range_low: 0.75
noise_range_high: 0.75
num_train_epochs: 2
global_batch_size: 8
micro_batch_size: 1
lr: 2.0e-6
lr_warmup_ratio: 0.03
lr_decay_style: cosine
weight_decay: 0.01
max_grad_norm: 1.0
enable_mixed_precision: true
enable_gradient_checkpointing: true
enable_full_shard: true
enable_fsdp_offload: true
ckpt_manager: dcp
save_hf_weights: true
```

Run both:

```text
Run A: 2 epochs, strict DMax-comparable setting.
Run B: 3–4 epochs, architecture-friendly setting.
```

The 2-epoch setting is the fair comparison. The 3–4 epoch setting tests whether T3 needs more optimisation.

---

## 4. DMax OPUT schedule to reproduce

DMax's OPUT data builder duplicates every sample into two versions:

```text
flag = False: normal masked noisy input
flag = True: on-policy predicted noisy input
```

The two versions are concatenated and shuffled, so the training mixture is approximately:

```text
50% masked noisy input
50% on-policy predicted noisy input
```

There is no epoch-wise curriculum in the released implementation.

### 4.1 Mask ratio

Use fixed mask ratio:

```text
mask ratio = 0.75
```

This means 75% of response-side maskable tokens are replaced by `[MASK]`.

### 4.2 `flag=False` branch

Input remains:

```text
25% clean visible response tokens
75% [MASK] response tokens
```

The model is trained to predict the clean target sequence.

### 4.3 `flag=True` branch — talk-only rollout with cached anchor

The released DMax training loop does:

1. No-grad full forward.
2. Argmax at masked positions.
3. Replace all masked positions with argmax predictions.
4. Grad forward on the updated input.
5. CE loss against the clean target.

**T3-DMax diverges here**, because:

- The think model has no LM head — only talk can produce token logits. So the rollout must invoke talk.
- T3's inference flow runs think once per block, then talk many times. The training loop should match that distribution: the anchor is computed once on the all-mask block, and talk does the iterative denoising. Recomputing think on the updated predicted input would be ~50% wasted compute, and would also train think on a distribution it never sees at inference.

Adapted procedure for `flag=True`:

```text
masked noisy input
→ no-grad think forward on masked input  ──┐
                                            │ save anchor (last-layer hidden state)
→ no-grad talk forward, conditioned on anchor
→ argmax over talk's logits at masked positions
→ replace all masked positions in input_ids with talk's argmax predictions

→ grad talk forward, REUSING the cached anchor, on the updated (predicted) input
→ CE loss against clean target
```

Notes:

- Think gets no gradient on `flag=True` samples. Think is only trained on `flag=False` (masked input) samples — which is the only distribution it sees at inference.
- Talk gets gradient on the updated input with the masked-input anchor, matching its inference-time distribution after the first iteration.
- No confidence thresholding in the first implementation. Replace all masked positions, same as DMax.

---

## 5. High-level architecture

Implement a model wrapper called something like:

```python
ThinkTalkLLaDA2ForMaskedDiffusionLM
```

It should expose a HuggingFace-like forward interface:

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    labels=None,
    use_cache=False,
    output_hidden_states=False,
    output_router_logits=False,
)
```

Return object must contain at least:

```python
outputs.logits
```

This is necessary because the DMax training loop expects:

```python
model(...).logits
```

---

## 6. Model components

### 6.1 Think model

The think model is the heavy pretrained dLLM backbone.

Milestone-1 default:

```text
think_model = LLaDA-2.0-mini (full, MoE-merged) — DMax baseline backbone, unpruned
train_think = true                              # full fine-tuning, matches DMax
prune_think_last_n_layer = 0
```

Ablations:

```text
A1.5: prune_think_last_n_layer = 1, talk warm-started with the removed layer
A3:   train_think = false (freeze; tests whether anchor quality from
      the pretrained backbone alone is sufficient)
```

Do not start from freeze-only or prune-only in milestone 1. The first run keeps full LLaDA-2.0-mini intact for direct comparison to DMax. Pruning is a separate hypothesis (Q-think-prune) tested in A1.5.

### 6.2 Talk model

The talk model is a lightweight decoder trained to perform iterative block denoising.

Milestone-1 defaults:

```text
talk_hidden_size = think_hidden_size
talk_num_layers = 2                   # matches current Think-Then-Talk
talk_num_attention_heads = match think (same hidden size, same head config)
talk_mlp_ratio = 4
talk_arch = dense transformer block, no MoE
reuse token embedding from think model
LM head: copy from think (trainable from this initialisation — see 6.5)
```

Why depth = 2 first:

- The current Think-Then-Talk codebase uses `n_layers=2` and `prune_last_n_layer=2`. Starting from the same depth means a single variable changed vs. the existing T3 codebase (backbone), and a single variable changed vs. DMax (think/talk split).
- The 4–6 layer convention from EAGLE / DFlash is a speculative-decoding heritage and is not validated for diffusion denoising. Treat it as an ablation knob.

Why same hidden size first:

- avoids extra projection bugs;
- allows direct reuse of embedding and LM head;
- makes the first experiment about depth reduction, not width mismatch.

Later ablations:

```text
talk_num_layers = 2 (default), 4, 6, 8
talk_hidden_size < think_hidden_size with input/output projectors
```

### 6.3 Anchor projection / fuser

**Milestone-1 default: last-layer hidden state only.**

```text
anchor = think.hidden_states[-1]      # [batch, seq_len, think_hidden_size]
```

No fusion, no projection, no RMSNorm — talk consumes the last-layer hidden state directly via the conditioning mechanism in 6.4. Since `talk_hidden_size == think_hidden_size` in milestone 1, no shape conversion is needed.

**The fuser code path must still be present and configurable**, because milestone-2 and later trials will ablate multi-layer fusion. The configuration field is:

```yaml
anchor_fuser_type: last_only          # default
# alternatives: last_only | last_mid | concat_linear | gated | cross_attention
anchor_layers: "last"                 # default; ignored when anchor_fuser_type=last_only
```

When `anchor_fuser_type != last_only`, instantiate the appropriate fuser module and apply:

```python
anchor = fuser(selected_hidden_states)    # [batch, seq_len, talk_hidden_size]
```

For all fuser types except `last_only`, the fuser output is RMSNorm'd.

The multi-layer concat (10/20/30) used in earlier trials and in EAGLE/DFlash is **not the milestone-1 default**. It is available as the `concat_linear` fuser type for later ablations. Note that "10, 20, 30" was hand-picked for LLaDA-8B's depth and will not generalise to LLaDA-2.0-mini's smaller layer count — when re-enabling multi-layer fuse, express anchor layers as ratios of `think_num_layers` (e.g. `L/3, 2L/3, L-1`).

### 6.4 Conditioning mechanism

Use a simple gated residual injection first:

```python
talk_hidden = talk_hidden + gate * anchor_proj(anchor)
```

Where `gate` is either:

```python
gate = sigmoid(W_gate(talk_hidden))
```

or a scalar parameter initialised near zero, e.g.:

```python
gate = sigmoid(alpha), alpha initialized to -2.0
```

This avoids overwhelming the newly initialized talk model at the beginning.

Later option:

```text
cross-attention from talk hidden states to anchor hidden states
```

Do not implement cross-attention first unless the simple version fails.

### 6.5 LM head

The think model has no token head — token logits are produced by talk's LM head.

Milestone-1 default: **copy** the LM head weights from LLaDA-2.0-mini into a fresh `nn.Parameter` attached to the talk model, and train it (`train_lm_head = true`). This gives talk a warm initialisation while still allowing it to adapt to the new conditioning signal.

The current Think-Then-Talk has both modes (`train_lm_head_enabled: false` aliases by reference, `true` clones into a trainable parameter). Keep both available; default to `true` here because the LM head sees talk outputs, not LLaDA's raw outputs, and may need to shift.

---

## 7. Forward pass design

### 7.1 Training forward

The training forward should accept the same `input_ids` that DMax passes in.

In block diffusion mode, DMax constructs:

```python
full_input_ids = torch.cat([noisy_input_ids, clean_input_ids], dim=1)
```

So sequence length is `2 * max_seq_len`.

Inside T3-DMax model:

```python
def forward(input_ids, attention_mask, position_ids, ...):
    # 1. Run think model
    think_outputs = think_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_hidden_states=True,
        use_cache=False,
    )

    # 2. Fuse selected hidden states into anchor
    anchor = anchor_fuser(think_outputs.hidden_states)

    # 3. Run talk model conditioned on anchor
    talk_outputs = talk_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        anchor=anchor,
    )

    # 4. Project to vocab logits
    logits = lm_head(talk_outputs.last_hidden_state)

    return CausalLMOutput(logits=logits)
```

Important: For the first implementation, compute logits for the whole doubled sequence and let the training script slice the noisy half, exactly like DMax.

### 7.2 Inference forward

For inference, the efficient version should not run the heavy model every talk iteration.

For each generated block:

```text
1. Current block starts as all [MASK].
2. Run think model once to produce anchor.
3. For step = 1..block_size:
   - run talk model with current tokens + static anchor
   - predict logits
   - reveal selected tokens
   - update current tokens
4. Stop when all tokens in block are decoded.
```

Do not implement full inference before the training script is stable.

---

## 8. Training script adaptation

Start from:

```text
dFactory/tasks/train_llada2_bd_oput.py
```

Create:

```text
dFactory/tasks/train_t3_dmax_bd_oput.py
```

Keep most of the original structure:

- distributed initialisation
- data loading
- `process_mdm_sft_example`
- block diffusion attention mask
- model building
- optimizer
- LR scheduler
- activation offloading
- checkpointing
- HF weight saving

Modify only the model construction and the OPUT rollout forward as needed.

### 8.1 New model args

Add model arguments:

```python
@dataclass
class T3ModelArguments(ModelArguments):
    attn_implementation: Optional[Literal["eager", "sdpa", "flex_attention"]] = "sdpa"
    think_model_path: str = ""
    talk_num_layers: int = 2
    talk_hidden_size: int = -1            # -1 = match think_hidden_size
    talk_num_attention_heads: int = -1    # -1 = match think
    anchor_fuser_type: str = "last_only"  # last_only | last_mid | concat_linear | gated | cross_attention
    anchor_layers: str = "last"           # ignored when anchor_fuser_type=last_only
    anchor_conditioning: str = "gated_residual"
    prune_think_last_n_layer: int = 0     # 0 = full think; >0 = warm-start talk with these layers (ablation A1.5)
    train_think: bool = True
    train_talk: bool = True
    train_lm_head: bool = True
```

### 8.2 New train args

Add:

```python
@dataclass
class T3TrainingArguments(LLaDA2TrainingArguments):
    t3_rollout_mode: Literal["dmax_oput", "none"] = "dmax_oput"
    t3_rollout_target: Literal["talk_only", "think_and_talk"] = "talk_only"
    t3_rollout_replace: Literal["all_masked", "confidence"] = "all_masked"
    t3_train_iterations: int = 1
```

For the milestone-1 run:

```yaml
t3_rollout_mode: dmax_oput
t3_rollout_target: talk_only        # think runs once on masked input; talk does the rollout
t3_rollout_replace: all_masked
t3_train_iterations: 1
```

`t3_rollout_target=talk_only` reuses the think anchor from the masked input across both the rollout forward and the grad forward, matching T3's inference compute pattern. See sec 8.3 for the loop.

`t3_train_iterations=1` means: one talk forward predicts all masked positions, matching DMax OPUT's single rollout step. Multi-step talk training (a T3-specific lever DMax does not have) is promoted to ablation A4 and is the highest-priority follow-up once milestone-1 lands.

### 8.3 OPUT branch — talk-only rollout

DMax's released logic is "no-grad full forward → argmax → replace → grad full forward." In T3-DMax we split this into a no-grad think forward (once, on masked input), a no-grad talk rollout, and a grad talk forward — reusing the cached anchor.

```python
if args.train.t3_rollout_mode == "dmax_oput" and micro_batch["flag"].item() is True:
    model.eval()
    with torch.no_grad():
        # 1. Think forward on masked input — produces the anchor we will reuse.
        think_hidden = model.run_think_model(
            input_ids=micro_batch["input_ids"],
            attention_mask=micro_batch["attention_mask"],
            position_ids=micro_batch["position_ids"],
            use_cache=False,
            output_hidden_states=True,
        )
        anchor = model.build_anchor(think_hidden)           # last-layer by default

        # 2. Talk forward, conditioned on anchor — produces rollout logits.
        rollout_logits = model.run_talk_model(
            input_ids=micro_batch["input_ids"],
            anchor=anchor,
            attention_mask=micro_batch["attention_mask"],
            position_ids=micro_batch["position_ids"],
        )

        # 3. Replace masked positions in the noisy half with talk argmax.
        rollout_tokens = rollout_logits.argmax(dim=-1)
        noisy_len = noisy_input_ids.shape[1]
        active_mask = micro_batch["input_ids"][:, :noisy_len] == mask_token_id
        micro_batch["input_ids"][:, :noisy_len] = torch.where(
            active_mask,
            rollout_tokens[:, :noisy_len],
            micro_batch["input_ids"][:, :noisy_len],
        )
    model.train()

    # 4. Grad forward: REUSE the cached anchor; only talk runs with gradient.
    logits = model.run_talk_model(
        input_ids=micro_batch["input_ids"],
        anchor=anchor.detach(),                              # frozen — anchor was no-grad
        attention_mask=micro_batch["attention_mask"],
        position_ids=micro_batch["position_ids"],
    )
else:
    # flag=False branch: standard full forward (think + talk), gradients flow into both.
    logits = model(**micro_batch, use_cache=False, output_router_logits=False).logits

noisy_logits = logits[:, :noisy_len].contiguous()
loss = CE(noisy_logits, labels)
```

Properties of this loop:

| Sample type | Think sees gradient | Talk sees gradient | Inputs talk sees |
|---|---|---|---|
| `flag=False` (masked) | yes | yes | masked input + masked-input anchor |
| `flag=True` (predicted) | **no** | yes | predicted input + masked-input anchor (cached) |

Think therefore only ever sees gradient on masked inputs, which is exactly the distribution it sees at inference (think runs once per block on the all-mask block). Talk sees both — matching its inference distribution after iteration 1+.

Compute per OPUT (`flag=True`) sample is roughly **1 think forward + 2 talk forwards**, vs. DMax's **2 full forwards** on a single backbone. With talk = 2 layers against LLaDA-2.0-mini's 20 layers, talk forwards are ~10% of a think forward; total compute is ~1.2× a single LLaDA-2.0-mini forward vs. DMax's 2.0× — roughly a **40% training-step compute reduction** for `flag=True` samples (about half the dataset). The training cost reduction is the point of the experiment; needs verification on the M0 reproduction run.

---

### 8.4 Anchor-leak verification (mandatory pre-training check)

DMax's block-diffusion mode constructs `cat([noisy, clean], dim=1)` and slices `[:, :noisy_len]` from the logits for the loss. The attention mask is **not** a blanket noisy→clean block — DMax intentionally lets a noisy query in block `i` attend to clean keys in **earlier blocks** (the offset-block-causal sub-mask `M_OBC`), because at inference time those earlier clean tokens represent positions the model has already decoded.

The actual leak invariant to verify is stricter:

> For a noisy query in block `i`, it must **not** attend to any clean key in block `i` (its own) or in any block `j > i` (future).

If a noisy query in block `i` could see clean tokens in its own block, those clean tokens **are** the labels — training would be a trivial copy task and inference would collapse. If it could see clean tokens in a future block, it leaks future labels.

The implemented test suite (`tests/test_anchor_leak.py`) covers:

1. **`test_noisy_cannot_see_own_or_future_clean_blocks`** (parametrized over `(seq_len, block_size)`): scans the mask returned by `block_diffusion_mask(...)` and asserts that for every `(noisy_q, clean_kv)` cell where `clean_block(kv) >= noisy_block(q)`, the mask is `False`. No model needed.
2. **`test_noisy_can_see_prior_clean_blocks`**: positive sanity — the offset-block-causal sub-mask must allow noisy queries in block `i` to see clean keys in blocks `< i`. If this fails the training distribution is wrong (no prior-context signal).
3. **`test_noisy_self_attention_is_block_diagonal`**: noisy queries in block `i` see only noisy keys in block `i`.
4. **`test_anchor_invariant_to_future_clean_perturbation`** (model-level, marked `slow`): perturb clean tokens only in block `i` of the clean half; verify the noisy anchor in block `i` is unchanged. Held to fp32 tolerance `< 1e-5`.
5. **`test_talk_logits_invariant_to_future_clean_perturbation`** (model-level, marked `slow`): same property for talk logits, tolerance `< 1e-4`.
6. **`test_anchor_DOES_change_when_prior_clean_block_perturbed`** (model-level, marked `slow`): positive sanity — perturb only clean block 0; the noisy anchor in block 1 must change (proving the mask exposes prior context as designed).

Run before any training session:

```bash
cd <T3-DMax repo>
PYTHONPATH=dFactory:dFactory/VeOmni:$PYTHONPATH pytest tests/test_anchor_leak.py -v --runslow
```

If any test fails, do not train. Fix the attention mask or the model first. This is a 1-day investment that prevents a multi-week debugging round.

---

## 9. Loss design

For the strict DMax-comparable run, keep DMax loss:

```python
loss = cross_entropy(
    noisy_logits.view(-1, vocab_size),
    labels.view(-1),
    reduction="none",
)

loss = loss.sum() / (labels != -100).sum()
```

Use:

```yaml
same_token_labels: true
```

Do not introduce multi-loss weighting in the first comparable run.

Later ablation:

```text
loss = reveal_token_loss
     + 0.3 * all_masked_aux_loss
     + 0.1 * final_block_loss
```

But this should be after the strict DMax-comparable baseline.

---

## 10. Config file

Create:

```text
dFactory/configs/sft/t3_llada2_mini_bd_oput.yaml
```

Initial content:

```yaml
model:
  config_path: ./configs/model_configs/think_talk_llada2_mini
  model_path: /path/to/LLaDA2.0-mini-moe-merge
  tokenizer_path: /path/to/LLaDA2.0-mini-moe-merge
  think_model_path: /path/to/LLaDA2.0-mini-moe-merge
  attn_implementation: sdpa
  moe_implementation: fused

  # T3-specific
  talk_num_layers: 2
  talk_hidden_size: -1                  # -1 = match think_hidden_size
  talk_num_attention_heads: -1          # -1 = match think
  anchor_fuser_type: last_only
  anchor_layers: "last"
  anchor_conditioning: gated_residual
  prune_think_last_n_layer: 0           # 0 = full think model
  train_think: true
  train_talk: true
  train_lm_head: true

data:
  train_path: ./my_data/postprocess_train.jsonl
  data_type: conversation
  datasets_type: mapping
  dataloader_type: native
  max_seq_len: 2048
  text_keys: messages
  noise_range_low: 0.75
  noise_range_high: 0.75
  num_workers: 16

train:
  output_dir: ./t3_llada2_mini_bd_oput_outputs
  data_parallel_mode: fsdp2
  tensor_parallel_size: 1
  ulysses_parallel_size: 1
  expert_parallel_size: 1

  global_batch_size: 8
  micro_batch_size: 1
  num_train_epochs: 2

  optimizer: adamw
  beta1: 0.9
  beta2: 0.999
  lr: 2.0e-6
  lr_warmup_ratio: 0.03
  lr_decay_style: cosine
  lr_decay_ratio: 1.0
  weight_decay: 0.01
  max_grad_norm: 1.0

  enable_mixed_precision: true
  enable_gradient_checkpointing: true
  enable_full_shard: true
  enable_fsdp_offload: true
  enable_activation_offload: false

  init_device: meta
  broadcast_model_weights_from_rank0: true
  enable_full_determinism: false
  empty_cache_steps: 500

  ckpt_manager: dcp
  load_checkpoint_path: ""
  save_epochs: 1
  save_hf_weights: true

  block_diffusion_mode: true
  block_size: 32
  same_token_labels: true

  # T3/OPUT
  t3_rollout_mode: dmax_oput
  t3_rollout_target: talk_only          # think runs once on masked input; talk does rollout
  t3_rollout_replace: all_masked
  t3_train_iterations: 1

  use_wandb: false
  log_steps: 1
```

---

## 11. Data preparation

Use DMax's public datasets.

Math:

```bash
cd dFactory
python scripts/build_dataset_oput.py \
  --dataset_path Zigeng/DMax-LLaDA-2.0-Mini-Math-Trajectories \
  --out_dir ./my_data \
  --seed 42
```

Code:

```bash
python scripts/build_dataset_oput.py \
  --dataset_path Zigeng/DMax-LLaDA-2.0-Mini-Code-Trajectories \
  --out_dir ./my_code_data \
  --seed 42
```

For the first experiment, use math only.

---

## 12. Weight preparation

Follow DMax:

```bash
cd dFactory

python scripts/download_hf_model.py \
  --repo_id inclusionAI/LLaDA2.0-mini \
  --local_dir /path/to/separate_expert_model

python scripts/moe_convertor.py \
  --input-path /path/to/separate_expert_model \
  --output-path /path/to/LLaDA2.0-mini-moe-merge \
  --mode merge
```

Use the merged model path in the T3-DMax config.

---

## 13. Training command

```bash
cd dFactory

PYTHONPATH=$(pwd)/VeOmni:$PYTHONPATH \
sh train.sh \
  tasks/train_t3_dmax_bd_oput.py \
  configs/sft/t3_llada2_mini_bd_oput.yaml
```

---

## 14. Evaluation plan

### 14.1 First evaluation

Before implementing efficient T3 inference, evaluate with a simple non-optimised generation loop:

```text
for each block:
    run think model once
    run talk model repeatedly up to block_size steps
    reveal tokens using DMax threshold or uniform strategy
```

Metrics:

```text
GSM8K strict accuracy
GSM8K flexible accuracy
tokens per forward (TPF)
number of forward passes
average revealed tokens per talk step
wrong high-confidence reveal rate
```

### 14.2 Reveal strategy

Start with DMax-style threshold reveal:

```text
reveal tokens whose confidence exceeds threshold
fallback: reveal at least one token
```

Then implement DMax-style uniform reveal:

```text
decode a confident left-to-right contiguous prefix
stop at first low-confidence masked token
fallback: reveal the leftmost masked token
```

### 14.3 Soft embedding SPD — talk-side only

DMax sends hybrid embeddings through the full LLaDA backbone every iteration. T3-DMax does **not**: think only ever runs once per block, on a fresh all-mask input. Soft embeddings flow into talk only.

Inference flow with SPD:

```text
For each block:
  1. Think forward on [prompt + all-mask block]              # once
  2. Build anchor from think.hidden_states[-1]                # once
  3. For step = 1..max_steps:
       a. Construct talk inputs:
            - mask positions: e_mask
            - token positions: hybrid embed = p * e_token + (1-p) * e_mask, then renorm
       b. Talk forward, conditioned on anchor (frozen)
       c. Compute argmax + confidence; promote longest contiguous
          confident prefix from mask → token (DMax's uniform reveal)
       d. Update mask/token sets and per-position p, e_token
  4. Stop when block fully decoded
```

Order of implementation:

1. Hard-token reveal with threshold first.
2. DMax uniform left-to-right reveal.
3. Top-1 SPD (talk side only).
4. Top-k SPD.

Normalize the talk-input embedding norm at the talk embedding's RMS to avoid norm collapse, the same way DMax does for its LLaDA inputs.

Because soft embeddings only traverse 2 talk layers (vs. all 20 layers of LLaDA-2.0-mini in DMax), SPD is structurally cheaper in T3-DMax for the same iteration count. This is one of the points where T3 should beat DMax at matched TPF.

---

## 15. Logging requirements

During training, log:

```text
training/loss
training/lr
training/grad_norm

t3/masked_branch_loss
t3/oput_branch_loss
t3/rollout_token_acc
t3/rollout_masked_positions
t3/rollout_wrong_high_confidence_rate

t3/anchor_norm
t3/talk_hidden_norm
t3/gate_mean
t3/gate_std
```

During evaluation, log:

```text
eval/gsm8k_strict_acc
eval/gsm8k_flexible_acc
eval/tpf
eval/avg_forwards_per_sample
eval/avg_revealed_tokens_per_step
eval/high_conf_wrong_rate
eval/eos_early_stop_rate
```

These metrics are essential. If final GSM8K is low, these logs will tell whether the issue is:

```text
anchor quality
talk model capacity
rollout mismatch
confidence calibration
bad reveal strategy
```

---

## 16. Ablation plan

Run in this order:

### A0. DMax reproduction

Train/run unmodified DMax on LLaDA-2.0-mini to confirm the environment. Expected GSM8K ≈ 90–92%, TPF ≈ 5.4.

### A1. T3-DMax milestone-1 baseline

```yaml
backbone: LLaDA-2.0-mini (full, unpruned)
talk_num_layers: 2
talk_hidden_size: match think
anchor_fuser_type: last_only
anchor_conditioning: gated_residual
t3_rollout_target: talk_only
t3_train_iterations: 1
block_size: 32
epochs: 2
data: DMax-LLaDA-2.0-Mini-Math-Trajectories
loss: uniform CE (no step weighting)
```

This is the only milestone-1 config. Everything else is an ablation against it.

### A1.5. Prune-and-transplant talk init

```yaml
prune_think_last_n_layer: 1
# talk model is initialised from LLaDA-2.0-mini's last transformer layer
# (one of talk's two layers) + one fresh layer
```

Tests whether warm-starting talk with LLaDA's denoising head helps. Direct comparison to A1 isolates the effect of init.

### A2. More epochs

```text
A2: 3 epochs, same everything else as A1
A2b: 4 epochs
```

### A3. Freeze think

```yaml
train_think: false
train_talk: true
```

Tests whether think actually needs to adapt or whether anchor quality from the pretrained backbone is sufficient.

### A4. Multi-step talk training (T3's differentiator vs. DMax)

```yaml
t3_train_iterations: 2, 4, 8
```

DMax has no analogue. This is the highest-priority follow-up after A1 because it directly tests whether T3's compute pattern (think once, talk many) gives the talk model a useful training distribution that single-step DMax cannot.

### A5. Talk depth

```text
talk_num_layers = 2 (A1), 4, 6, 8
```

The 4–6 layer convention from EAGLE/DFlash is tested here, not assumed.

### A6. Anchor fuser

```text
A6a: last_only (A1 default)
A6b: last_mid              # last + middle
A6c: concat_linear         # h_{L/3}, h_{2L/3}, h_{L-1}, fused with linear
A6d: gated                 # learned gates per layer
A6e: cross_attention       # talk attends to all selected think layers
```

Anchor layer indices must be expressed as ratios of `think_num_layers`, not hardcoded — LLaDA-2.0-mini does not have 30 layers.

### A7. Inference reveal strategy

```text
A7a: threshold reveal
A7b: DMax uniform left-to-right reveal
A7c: top-1 SPD (talk-side only)
A7d: top-k SPD
```

### A8. Talk model conditioning

```text
A8a: gated_residual (A1 default)
A8b: cross_attention talk→anchor
A8c: anchor prepended as a soft prefix token
```

---

## 17. Expected implementation risks

### 17.1 Memory

Full fine-tuning with both think and talk models is heavier than DMax. If OOM happens:

1. reduce `micro_batch_size`;
2. enable activation offload;
3. reduce talk layers;
4. temporarily freeze think for debugging;
5. use smaller `max_seq_len` for smoke tests.

### 17.2 Model registry

DMax uses VeOmni model registry. The new model must be registered similarly to DMax's `models.llada2_moe`.

Add in the training script:

```python
from veomni.models.registry import ModelRegistry
ModelRegistry.register_modeling_path("models.think_talk_llada2")
```

### 17.3 Output compatibility

The model output must expose:

```python
.logits
```

Otherwise the DMax training loop will break.

### 17.4 Checkpoint saving

Make sure the HF checkpoint saves:

```text
think model weights
talk model weights
anchor fuser weights (if anchor_fuser_type != last_only — last_only has no params)
talk LM head weights (if train_lm_head=true)
gating parameters from anchor_conditioning
config (including all T3-specific fields)
tokenizer
```

The config must record T3-specific fields so that inference scripts can rebuild the model with the right talk depth, fuser type, conditioning mode, etc.

### 17.5 MoE merge format

If the think model uses LLaDA2.0-mini MoE, use the merged-expert format for training.

Do not accidentally load the separate-expert checkpoint into the fused MoE training implementation.

---

## 18. Done criteria for the coding agent

The implementation is complete when:

1. `train_t3_dmax_bd_oput.py` launches with the new config.
2. A 10-step smoke test runs without crashing.
3. Loss decreases on a small subset.
4. HF checkpoint export works.
5. A simple generation script can load the checkpoint.
6. GSM8K evaluation runs end-to-end.
7. Logs include rollout accuracy, TPF, and reveal statistics.

---

## 19. Milestone breakdown

### Milestone 0 — environment + leak verification (1–2 days)

```text
- DMax repo cloned as base.
- LLaDA-2.0-mini downloaded and MoE-merged.
- DMax A0 reproduction runs end-to-end (10 training steps + 1 GSM8K eval batch).
- Sec 8.4 anchor-leak verification unit test passes on DMax's attention mask.
```

### Milestone 1 — strict A1 baseline (3–5 days)

```text
- New ThinkTalk model class registered in VeOmni.
- Talk model = 2-layer dense transformer, same hidden as think.
- Anchor = think.hidden_states[-1] (last-layer only).
- LM head = copied from LLaDA-2.0-mini, trainable.
- Full fine-tuning of think + talk + LM head.
- DMax OPUT script adapted with t3_rollout_target=talk_only.
- 100 training steps run without crash.
- HF checkpoint export + reload works.
- Simple GSM8K eval with threshold reveal runs end-to-end.
```

### Milestone 2 — full A1 run + first ablation (1–2 weeks)

```text
- 2-epoch math training on full DMax math dataset.
- GSM8K eval at threshold + uniform reveal.
- A1.5 (prune-and-transplant) launched in parallel.
```

### Milestone 3 — T3 differentiator (A4 multi-step talk training)

Only enter this once A1 ≥ ~85% GSM8K with TPF ≥ 3.5. If A1 is below those thresholds, debug A1 before adding new variables.

---

## 20. Main principle

Do not change too many things at once.

The milestone-1 experiment differs from DMax in exactly two major ways:

```text
DMax:
    one full LLaDA-2.0-mini model
    every diffusion step runs the full backbone

T3-DMax milestone 1:
    full LLaDA-2.0-mini as think; 2-layer talk model
    every diffusion step at inference runs only talk
    OPUT rollout uses talk only; think anchor cached
```

Everything else (data, block size, mask ratio, LR, epochs, optimizer, FSDP config) stays as close to DMax as possible. That is the cleanest way to test H1 — that the think/talk split is a real architecture improvement, not a hyperparameter difference.

The single most important variable to **not** vary in milestone 1: loss formulation. Stay with uniform CE. Step-weighted loss is ablation territory, not baseline territory.
