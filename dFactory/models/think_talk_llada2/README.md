# `think_talk_llada2`

Model definition for T3-D milestone 1.

| File | What it holds |
|---|---|
| `configuration_think_talk_llada2.py` | `ThinkTalkLLaDA2Config` — extends `LLaDA2MoeConfig` with `talk_*`, `anchor_*`, `prune_think_*` fields. |
| `modeling_think_talk_llada2.py` | `ThinkTalkLLaDA2ForCausalLM`, `TalkModel`, `TalkDecoderLayer`, `AnchorFuser`, `GatedResidualConditioning`. |
| `__init__.py` | Exports `ModelClass = ThinkTalkLLaDA2ForCausalLM` for VeOmni's registry. |

## Forward in three lines

```python
anchor = think_model(input_ids).hidden_states[-1]        # full LLaDA-2.0-mini
talk_h = talk_model(embed(input_ids), anchor=anchor)      # 2-layer dense, gated injection at L0
logits = lm_head(talk_h)
```

Block-diffusion convention: when `block_diffusion_mode=true`, the training script passes
`cat([noisy, clean], dim=1)` as `input_ids`. The attention mask (built in the training
script) prevents the noisy half from attending to the clean half at every layer of both
think and talk. **Anchor leak is mandatory to verify** — see `tests/test_anchor_leak.py`.

## Reuse from DMax (Apache-2.0)

Imports from `models.llada2_moe.modeling_llada2_moe`:
- `LLaDA2MoeModel` -- entire think backbone, unmodified
- `ATTENTION_CLASSES` -- attention impls (sdpa / flash_attention_2 / eager)
- `LLaDA2MoeMLP` -- dense MLP used by talk decoder layers
- `LLaDA2MoeRMSNorm` -- norm impl
- `LLaDA2MoeRotaryEmbedding` -- RoPE (shared instance with think to guarantee identical basis)
- `LLaDA2MoePreTrainedModel` -- base class (inherits `_init_weights`, FSDP plan, etc.)

## Reuse from Think-Then-Talk (vendored patterns)

- Anchor injection at the start of talk layer 0 (the `LLaDAFuseBlock` pattern in
  `model/modeling_t3.py:L491,L757-L770`), simplified to scalar-gate + RMSNorm (no
  `rps_mlp_in/out` MLP, no learnable `eta`).
- `_prune_think_last_n_layer` follows `T3Model.prune_llada_last_n_blocks`
  (`model/modeling_t3.py:L942-L972`).
- Trainable LM head clone with init from word embeddings: `T3Model.__init__`
  branch on `train_lm_head_enabled` (`model/modeling_t3.py:L913-L921`).

## Things deliberately left out of milestone 1

- Multi-layer anchor fuser (`last_mid` / `concat_linear` / `gated` / `cross_attention`):
  scaffolded but not used. Default is `last_only`.
- Cross-attention conditioning: scaffolded in config (`anchor_conditioning`), not
  implemented in modeling.
- Soft-embedding inputs (SPD) for talk: training-time soft inputs are not used; inference
  will introduce them once hard-token reveal works (brief sec 14.3).
- Multi-step talk training (`t3_train_iterations > 1`): T3's main differentiator vs. DMax;
  intentionally parked in the training script as a no-op for milestone 1.
