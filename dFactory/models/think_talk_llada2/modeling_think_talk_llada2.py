# Copyright 2026 University of Sydney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Portions of this file are derived from:
#   - DMax (https://github.com/czg1225/DMax), Apache-2.0
#     Specifically: re-uses `LLaDA2MoeModel`, `LLaDA2MoeMLP`, `LLaDA2MoeRMSNorm`,
#     `LLaDA2MoeRotaryEmbedding`, and `ATTENTION_CLASSES` from
#     `dFactory/models/llada2_moe/modeling_llada2_moe.py`.
#   - Think-Then-Talk (internal, University of Sydney)
#     Patterns adapted: gated residual anchor injection at talk layer 0,
#     `prune_last_n_layer` think-side helper, trainable LM head clone.

"""Think-Then-Talk model built on LLaDA-2.0-mini (T3-D milestone 1).

Forward flow (training, block_diffusion_mode=True, doubled sequence):

    input_ids [B, 2L]  (cat([noisy, clean]))
        │
        ▼
    think_model = LLaDA2MoeModel (full LLaDA-2.0-mini, MoE intact)
        │  output_hidden_states=True
        ▼
    anchor_fuser (default: last_only -> hidden_states[-1])
        │
        ▼
    talk_model (T3-D-specific dense transformer, talk_num_layers)
        │  layer 0 injects anchor via gated residual:
        │    h += sigmoid(alpha) * RMSNorm(anchor)
        ▼
    lm_head  (trainable, initialised from LLaDA2 word embeddings)
        │
        ▼
    logits  [B, 2L, V]   -> training script slices [:, :L] for CE loss

Inference flow is the same except think runs once per block on [prompt + all-mask block]
and talk iterates with the anchor held constant. See the brief sec 7.2.

At inference time, the relative compute saving is `talk_layers / (think_layers + talk_layers)`
per iteration after the first. For `talk_num_layers=2` against LLaDA-2.0-mini's 20 dense+MoE
layers, that is ~9% of full-model compute per additional talk iteration (assuming the MoE
gating cost is dominated by the routed FFN — true with `num_experts_per_tok=8` and
`moe_intermediate_size=512`).
"""

from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

# DMax-vendored building blocks
from models.llada2_moe.modeling_llada2_moe import (  # type: ignore[import-not-found]
    ATTENTION_CLASSES,
    LLaDA2MoeMLP,
    LLaDA2MoeModel,
    LLaDA2MoePreTrainedModel,
    LLaDA2MoeRMSNorm,
    LLaDA2MoeRotaryEmbedding,
)

from .configuration_think_talk_llada2 import ThinkTalkLLaDA2Config


# ============================================================================
#                              Anchor fuser
# ============================================================================

class AnchorFuser(nn.Module):
    """Combines selected think-side hidden states into a single anchor tensor.

    Milestone-1 default (`last_only`) has zero learnable parameters: it simply returns
    `hidden_states[selected_layer_idx]`. Other fuser types are scaffolded but raise
    NotImplementedError until ablations A6.* are reached.
    """

    def __init__(self, config: ThinkTalkLLaDA2Config):
        super().__init__()
        self.config = config
        self.fuser_type = config.anchor_fuser_type
        self.anchor_indices = config.resolved_anchor_layer_indices()

        if self.fuser_type == "last_only":
            assert len(self.anchor_indices) == 1, (
                f"last_only fuser expects 1 layer index, got {self.anchor_indices}"
            )
            self.fuser_proj = None
            self.fuser_norm = None
        elif self.fuser_type in ("last_mid", "concat_linear"):
            n_in = len(self.anchor_indices)
            self.fuser_proj = nn.Linear(
                n_in * config.hidden_size,
                config.resolved_talk_hidden_size,
                bias=False,
            )
            self.fuser_norm = LLaDA2MoeRMSNorm(
                config.resolved_talk_hidden_size, eps=config.rms_norm_eps,
            )
        else:
            # gated / cross_attention -- reserved for A6.d / A6.e
            raise NotImplementedError(
                f"anchor_fuser_type={self.fuser_type} is reserved for future ablation"
            )

    def forward(self, hidden_states_tuple: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Args:
            hidden_states_tuple: think_model.hidden_states (length = think_num_layers + 1).
        Returns:
            anchor: [batch, seq_len, talk_hidden_size]
        """
        if self.fuser_type == "last_only":
            return hidden_states_tuple[self.anchor_indices[0] + 1]  # +1: HF returns embeddings + N layers

        selected = [hidden_states_tuple[i + 1] for i in self.anchor_indices]
        concat = torch.cat(selected, dim=-1)
        anchor = self.fuser_proj(concat)
        anchor = self.fuser_norm(anchor)
        return anchor


# ============================================================================
#                       Anchor conditioning (talk layer 0)
# ============================================================================

class GatedResidualConditioning(nn.Module):
    """Adds anchor into talk hidden states at the very start of layer 0.

        talk_hidden = talk_hidden + sigmoid(alpha) * RMSNorm(anchor)

    Where `alpha` is a scalar learnable parameter, initialised so sigmoid(alpha) is small
    (default sigmoid(-2.0) ≈ 0.12). This keeps the freshly initialised talk model close to
    its embedding-only behaviour at start of training; the gate opens as needed.

    Pattern vendored from Think-Then-Talk's RPS residual injection. We simplify: no
    rps_mlp_in/out projection, no learnable eta — just the scalar gate.
    """

    def __init__(self, config: ThinkTalkLLaDA2Config):
        super().__init__()
        self.anchor_norm = LLaDA2MoeRMSNorm(
            config.resolved_talk_hidden_size, eps=config.rms_norm_eps,
        )
        self.alpha = nn.Parameter(torch.tensor(float(config.anchor_gate_init)))

    def forward(self, talk_hidden: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.alpha)
        return talk_hidden + gate * self.anchor_norm(anchor)


# ============================================================================
#                          Talk decoder layer
# ============================================================================

class TalkDecoderLayer(nn.Module):
    """Single talk transformer block: dense MLP, LLaDA-2.0 attention, optional anchor
    injection at the *very first* layer only.

    Built from LLaDA-2.0 primitives (attention class + RMSNorm + dense MLP) so that
    initialisation, RoPE, attention-bias contract, and SDPA paths all match the think
    backbone byte-for-byte. The only T3-D-specific addition is the anchor conditioning
    module passed in at layer 0.
    """

    def __init__(
        self,
        config: ThinkTalkLLaDA2Config,
        layer_idx: int,
        anchor_conditioning: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.hidden_size = config.resolved_talk_hidden_size
        self.layer_idx = layer_idx

        # Reuse LLaDA-2.0's attention class for binary-compatible RoPE / SDPA / flex_attn.
        self.attention = ATTENTION_CLASSES[config._attn_implementation](
            config=config, layer_idx=layer_idx,
        )
        self.mlp = LLaDA2MoeMLP(config=config, intermediate_size=config.resolved_talk_intermediate_size)
        self.input_layernorm = LLaDA2MoeRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LLaDA2MoeRMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # Only layer 0 holds the anchor conditioning module; later layers have anchor_conditioning=None.
        self.anchor_conditioning = anchor_conditioning

    def forward(
        self,
        hidden_states: torch.Tensor,
        anchor: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value=None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # T3-D: anchor injection at the start of layer 0
        if self.anchor_conditioning is not None:
            if anchor is None:
                raise RuntimeError(
                    "anchor must be provided to TalkDecoderLayer with anchor_conditioning set "
                    "(this is layer 0)."
                )
            hidden_states = self.anchor_conditioning(hidden_states, anchor)

        # Standard LLaDA-2.0 decoder block, dense MLP variant.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, _, present_kv = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states.to(residual.device)

        return hidden_states


# ============================================================================
#                              Talk model
# ============================================================================

class TalkModel(nn.Module):
    """Lightweight transformer stack that performs iterative block denoising conditioned
    on a per-block anchor from the think model.

    Position embeddings: shares the think model's RoPE instance (passed in at construction)
    to guarantee identical RoPE basis and avoid double-init issues.
    """

    def __init__(self, config: ThinkTalkLLaDA2Config, rotary_emb: nn.Module):
        super().__init__()
        self.config = config
        self.rotary_emb = rotary_emb  # shared with think; no copy

        anchor_conditioning = GatedResidualConditioning(config)
        self.layers = nn.ModuleList(
            [
                TalkDecoderLayer(
                    config,
                    layer_idx=i,
                    anchor_conditioning=anchor_conditioning if i == 0 else None,
                )
                for i in range(config.talk_num_layers)
            ]
        )
        self.norm = LLaDA2MoeRMSNorm(config.resolved_talk_hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        anchor: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        # Build position embeddings (cos, sin) once and pass into every layer.
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                anchor=anchor if i == 0 else None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            )

        return self.norm(hidden_states)


# ============================================================================
#                              ForCausalLM
# ============================================================================

class ThinkTalkLLaDA2ForCausalLM(LLaDA2MoePreTrainedModel):
    """Outer wrapper exposing a HuggingFace-compatible `forward(...).logits` interface.

    `_no_split_modules` includes both think and talk decoder layer classes so FSDP2 can
    shard them as basic units.
    """

    config_class = ThinkTalkLLaDA2Config
    _no_split_modules = ["LLaDA2MoeDecoderLayer", "TalkDecoderLayer"]

    def __init__(self, config: ThinkTalkLLaDA2Config):
        super().__init__(config)
        self.config = config

        # ---- Think model: full LLaDA-2.0-mini (MoE intact) -----------------------
        # We optionally drop the last N transformer layers when running ablation A1.5.
        self.model = LLaDA2MoeModel(config)
        if config.prune_think_last_n_layer > 0:
            self._prune_think_last_n_layer(config.prune_think_last_n_layer)
            # (For A1.5, the user will warm-start `talk_model.layers[0]` with the removed
            # layer's weights via a separate checkpoint-loading hook; not done here.)

        # ---- Anchor fuser (last_only by default) -------------------------------
        self.anchor_fuser = AnchorFuser(config)

        # ---- Talk model -------------------------------------------------------
        # Share the rotary embedding instance with think to guarantee identical RoPE.
        self.talk_model = TalkModel(config, rotary_emb=self.model.rotary_emb)

        # ---- LM head (trainable, initialised from think's word embeddings) -----
        # LLaDA-2.0-mini ties its lm_head to word_embeddings. We always make a separate
        # trainable Linear here so the head can adapt to talk outputs. The weight is
        # initialised to match LLaDA-2.0's tied weight at load time (handled by VeOmni's
        # weight loader; if no weight ships, normal init is used).
        self.lm_head = nn.Linear(
            config.resolved_talk_hidden_size, config.vocab_size, bias=False,
        )

        # ---- Freeze / unfreeze ------------------------------------------------
        self._apply_train_flags()

        # Post init
        self.post_init()

    # -------------------------------------------------------------------- ablations

    def _prune_think_last_n_layer(self, n: int) -> None:
        """Removes the last N LLaDA2 decoder layers from think.

        Pattern from Think-Then-Talk's `prune_llada_last_n_blocks` (modeling_t3.py:L942).
        For ablation A1.5 the removed layer(s) should be transplanted into talk; that
        warm-start is wired up by an external checkpoint hook, not here.
        """
        if n <= 0:
            return
        layers = self.model.layers
        new_len = len(layers) - n
        assert new_len >= 1, f"cannot prune {n} layers from {len(layers)}"
        self.model.layers = nn.ModuleList(list(layers[:new_len]))
        # Update config bookkeeping so downstream code sees the new depth.
        self.config.num_hidden_layers = new_len
        self.model.config.num_hidden_layers = new_len

    def _apply_train_flags(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = self.config.train_think
        for p in self.talk_model.parameters():
            p.requires_grad = self.config.train_talk
        for p in self.lm_head.parameters():
            p.requires_grad = self.config.train_lm_head

    # -------------------------------------------------------------------- forward

    def get_input_embeddings(self) -> nn.Module:
        return self.model.word_embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.word_embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # 1. Think -- run full LLaDA-2.0-mini, surface all hidden states for the fuser.
        think_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            output_router_logits=False,
            return_dict=True,
        )
        # 2. Anchor fuser (last_only by default).
        anchor = self.anchor_fuser(think_out.hidden_states)

        # 3. Talk model. Shares word embeddings with think (no separate embed table).
        if inputs_embeds is None:
            talk_embeds = self.model.word_embeddings(input_ids)
        else:
            talk_embeds = inputs_embeds

        talk_hidden = self.talk_model(
            inputs_embeds=talk_embeds,
            anchor=anchor,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # 4. LM head
        logits = self.lm_head(talk_hidden)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    # -------------------------------------------------------------------- helpers

    @torch.no_grad()
    def run_think_and_anchor(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Forward through think + anchor fuser only. Used by the talk-only OPUT rollout
        (brief sec 8.3) and by efficient inference (think once per block)."""
        think_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            output_router_logits=False,
            return_dict=True,
        )
        return self.anchor_fuser(think_out.hidden_states)

    def run_talk(
        self,
        input_ids: torch.LongTensor,
        anchor: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Forward through talk + LM head only, with a pre-computed anchor. Returns logits."""
        talk_embeds = self.model.word_embeddings(input_ids)
        talk_hidden = self.talk_model(
            inputs_embeds=talk_embeds,
            anchor=anchor,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return self.lm_head(talk_hidden)


# VeOmni's ModelRegistry walks submodules of a registered package (not the package's
# __init__.py) looking for a module-level `ModelClass` attribute. DMax's modeling_llada2_moe.py
# does the same thing (line ~1575). Without this line, `register_modeling_path("models.think_talk_llada2")`
# silently registers nothing -> the loader falls back to HF AutoModel -> HF rejects the
# unknown model_type. See VeOmni/veomni/models/registry.py:57-78.
ModelClass = ThinkTalkLLaDA2ForCausalLM
