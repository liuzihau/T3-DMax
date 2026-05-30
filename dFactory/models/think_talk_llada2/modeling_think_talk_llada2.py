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

import math
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
    repeat_kv,
    rotate_half,
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

        talk_hidden = talk_hidden + gate * RMSNorm(anchor)

    Two modes for `gate`:

    - **Learnable (default)**: `gate = sigmoid(alpha)` where alpha is an `nn.Parameter`
      initialised from `config.anchor_gate_init` (a sigmoid input — set to -2.0 for
      gate ≈ 0.12 at init). The gate opens (alpha grows) as the optimiser finds it
      useful.

    - **Fixed**: `gate = config.anchor_gate_value` (a scalar buffer, not trainable).
      Use this to remove the gate as a variable when diagnosing other issues — talk
      always sees this fraction of the anchor, no co-adaptation between gate and talk.

    Pattern vendored from Think-Then-Talk's RPS residual injection. We simplify: no
    rps_mlp_in/out projection, no learnable eta — just the scalar gate.
    """

    def __init__(self, config: ThinkTalkLLaDA2Config):
        super().__init__()
        self.anchor_norm = LLaDA2MoeRMSNorm(
            config.resolved_talk_hidden_size, eps=config.rms_norm_eps,
        )
        self.learnable = bool(getattr(config, "anchor_gate_learnable", True))
        if self.learnable:
            self.alpha = nn.Parameter(torch.tensor(float(config.anchor_gate_init)))
            self.register_buffer("fixed_gate", torch.tensor(0.0), persistent=False)
        else:
            # Fixed-gate mode: register_buffer so the value travels with the module on
            # .to() / .cuda() but isn't picked up by the optimiser.
            self.alpha = None
            value = float(getattr(config, "anchor_gate_value", 0.2))
            self.register_buffer("fixed_gate", torch.tensor(value), persistent=False)

    def forward(self, talk_hidden: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.alpha) if self.learnable else self.fixed_gate
        return talk_hidden + gate * self.anchor_norm(anchor)

    @property
    def gate_value(self) -> float:
        """For logging: returns the current scalar gate magnitude (post-sigmoid if learnable)."""
        if self.learnable:
            with torch.no_grad():
                return torch.sigmoid(self.alpha).item()
        return float(self.fixed_gate.item())


# ============================================================================
#                       Cross-attention (hybrid_xattn mode)
# ============================================================================

def _apply_rotary_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                         unsqueeze_dim: int = 1) -> torch.Tensor:
    """Apply RoPE rotation to a single tensor (Q or K, not both at once).

    Args:
        x:   [B, num_heads, seq_len, head_dim]
        cos: [B, seq_len, rope_dim]
        sin: [B, seq_len, rope_dim]
    Returns:
        rotated x, same shape.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return torch.cat([x_embed, x_pass], dim=-1)


class TalkCrossAttention(nn.Module):
    """Cross-attention from talk's noisy stream (Q) to the think anchor (K, V).

    Sequence shape contract (in `hybrid_xattn` mode):
      - hidden_states (Q):  [B, L,  H]   talk's noisy positions
      - kv_states      :    [B, 2L, H]   anchor at [noisy_positions, clean_positions]
      - attention_mask :    [B, 1, L, 2L] additive (0 allowed, -inf blocked)
      - q_pos_emb      :    (cos_q, sin_q) for Q positions 0..L-1
      - kv_pos_emb     :    (cos_kv, sin_kv) for KV positions [0..L-1, 0..L-1]

    Architecture mirrors LLaDA2MoeSdpaAttention: GQA, RMSNorm on per-head Q/K,
    partial-rotary RoPE, separate output projection. Differences vs the LLaDA2
    self-attn block:
      1. Separate `q_proj` and `kv_proj` linears (because Q and K/V come from
         different streams with different lengths).
      2. RoPE is applied independently to Q and K so each uses its own position_ids.
      3. No KV cache: cross-attn KV is always the full anchor, never extends.
    """

    def __init__(self, config: ThinkTalkLLaDA2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.resolved_talk_hidden_size
        self.num_heads = config.resolved_talk_num_attention_heads
        self.head_dim = config.head_dim or self.hidden_size // self.num_heads
        partial_rotary_factor = (
            config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        )
        self.rope_dim = int(self.head_dim * partial_rotary_factor)
        self.num_key_value_heads = config.resolved_talk_num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Q projection (mirrors LLaDA2's `query_key_value` split for Q only).
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.use_qkv_bias,
        )
        # Combined KV projection (matches LLaDA2's GQA layout for K+V).
        self.kv_proj = nn.Linear(
            self.hidden_size,
            2 * self.num_key_value_heads * self.head_dim,
            bias=config.use_qkv_bias,
        )
        self.query_layernorm = LLaDA2MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = LLaDA2MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.dense = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        q_pos_emb: Tuple[torch.Tensor, torch.Tensor],
        kv_pos_emb: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        kv_len = kv_states.size(1)

        # Project Q.
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Project KV (combined linear -> split).
        kv = self.kv_proj(kv_states)
        kv = kv.view(bsz, kv_len, 2 * self.num_key_value_heads, self.head_dim)
        k, v = kv.split([self.num_key_value_heads, self.num_key_value_heads], dim=-2)
        k = k.transpose(1, 2)   # [B, H_kv, kv_len, D]
        v = v.transpose(1, 2)

        # Per-head RMSNorm on Q and K (LLaDA2 convention).
        q = self.query_layernorm(q)
        k = self.key_layernorm(k)

        # RoPE on Q and K independently (different position_ids).
        cos_q, sin_q = q_pos_emb
        cos_kv, sin_kv = kv_pos_emb
        q = _apply_rotary_single(q, cos_q, sin_q)
        k = _apply_rotary_single(k, cos_kv, sin_kv)

        # GQA: repeat K, V to match Q's head count.
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # SDPA needs contiguous inputs on CUDA when a custom mask is supplied.
        if q.device.type == "cuda" and attention_mask is not None:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_out = self.dense(attn_out)
        return attn_out


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
        enable_cross_attention: bool = False,
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

        # T3-D ADDED: cross-attention block (hybrid_xattn mode).
        # When enabled, every talk layer cross-attends from its noisy Q stream to the
        # full think anchor [B, 2L, H]. Placement is T5-decoder style: between self-attn
        # and MLP, with its own pre-attention layernorm and residual.
        if enable_cross_attention:
            self.pre_cross_attention_layernorm = LLaDA2MoeRMSNorm(
                self.hidden_size, eps=config.rms_norm_eps,
            )
            self.cross_attention = TalkCrossAttention(config=config, layer_idx=layer_idx)
        else:
            self.pre_cross_attention_layernorm = None
            self.cross_attention = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        anchor: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value=None,
        use_cache: bool = False,
        # T3-D ADDED: cross-attention kwargs (hybrid_xattn mode).
        anchor_kv: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # T3-D: anchor injection at the start of layer 0 (gated_residual / hybrid_xattn modes)
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

        # T3-D ADDED: cross-attention to the think anchor (hybrid_xattn mode).
        # Forces every talk layer to consume the anchor, complementing the gated-residual
        # injection at layer 0. Q comes from the noisy stream (L tokens); KV is the full
        # anchor (2L tokens = think's last-layer hidden at noisy + clean positions).
        if self.cross_attention is not None:
            if anchor_kv is None or cross_position_embeddings is None:
                raise RuntimeError(
                    "TalkDecoderLayer with cross_attention enabled requires anchor_kv and "
                    "cross_position_embeddings (hybrid_xattn mode)."
                )
            residual = hidden_states
            hidden_states = self.pre_cross_attention_layernorm(hidden_states)
            cross_out = self.cross_attention(
                hidden_states=hidden_states,
                kv_states=anchor_kv,
                attention_mask=cross_attention_mask,
                q_pos_emb=position_embeddings,
                kv_pos_emb=cross_position_embeddings,
            )
            hidden_states = residual + cross_out

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

        # T3-D injection modes:
        #   anchor_injection_mode="gated_residual":
        #     Anchor is added per-position into talk's residual stream.
        #     inject_layers="first" -> only layer 0 conditioning module; "all" -> every layer.
        #     Talk sequence length: 2L (noisy + clean).
        #   anchor_injection_mode="concat_segment":
        #     Anchor is concatenated into the input sequence as separate tokens.
        #     No GatedResidualConditioning modules are instantiated; talk attends to anchor
        #     tokens directly via self-attention. The 3L sequence assembly happens in the
        #     outer ThinkTalkLLaDA2ForCausalLM.forward; TalkModel just consumes hidden_states.
        #     Talk sequence length: 3L (noisy + anchor + clean).
        #   anchor_injection_mode="hybrid_xattn":
        #     Layer 0 receives anchor via gated residual (v2-style); EVERY layer also runs
        #     a cross-attention block from its noisy Q stream to the full think anchor
        #     [B, 2L, H] as K/V. The clean stream is not in talk's sequence at all -- it
        #     reaches talk only as cross-attn K/V (with a block-causal mask).
        #     Talk sequence length: L (noisy only).
        self.injection_mode = config.anchor_injection_mode
        self.inject_all = (config.anchor_inject_layers == "all")

        if self.injection_mode == "gated_residual":
            has_residual_inject = lambda i: (i == 0 or self.inject_all)
            has_cross_attention = lambda i: False
        elif self.injection_mode == "concat_segment":
            has_residual_inject = lambda i: False
            has_cross_attention = lambda i: False
        elif self.injection_mode == "hybrid_xattn":
            # v2-style gated residual at the configured layer(s), cross-attn everywhere.
            has_residual_inject = lambda i: (i == 0 or self.inject_all)
            has_cross_attention = lambda i: True
        else:
            raise ValueError(f"Unknown anchor_injection_mode: {self.injection_mode}")

        self.layers = nn.ModuleList(
            [
                TalkDecoderLayer(
                    config,
                    layer_idx=i,
                    anchor_conditioning=(
                        GatedResidualConditioning(config) if has_residual_inject(i) else None
                    ),
                    enable_cross_attention=has_cross_attention(i),
                )
                for i in range(config.talk_num_layers)
            ]
        )

        self.norm = LLaDA2MoeRMSNorm(config.resolved_talk_hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        anchor: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
        # T3-D ADDED: cross-attention inputs (hybrid_xattn mode).
        anchor_kv: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Mode-specific contract:

        gated_residual:
            inputs_embeds [B, 2L, D]; anchor [B, 2L, D] mixed into residual at configured layers.
            anchor_kv / cross_attention_mask / cross_position_ids: not used.
        concat_segment:
            inputs_embeds [B, 3L, D] (already segment-embedded); anchor=None.
            cross-attn args: not used.
        hybrid_xattn:
            inputs_embeds [B, L, D] (noisy stream only); anchor [B, 2L, D] is used in two
            places: (a) sliced anchor[:, :L, :] for the gated-residual injection at layer 0
            (the caller passes the sliced tensor), (b) full anchor_kv [B, 2L, D] for the
            per-layer cross-attention.
        """
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        # T3-D ADDED: compute cross-attn position embeddings once (KV positions, 2L long).
        # The 2L K/V positions are [0..L-1, 0..L-1] (DMax parallel-position convention),
        # passed in via cross_position_ids. rotary_emb just indexes into its precomputed
        # tables by position_ids, so the same instance handles any positions.
        if anchor_kv is not None and cross_position_ids is not None:
            cross_position_embeddings = self.rotary_emb(anchor_kv, cross_position_ids)
        else:
            cross_position_embeddings = None

        hidden_states = inputs_embeds
        for i, layer in enumerate(self.layers):
            # anchor (for gated residual) is only consumed when the layer has a conditioning module.
            layer_anchor = (
                anchor
                if (
                    self.injection_mode in ("gated_residual", "hybrid_xattn")
                    and (i == 0 or self.inject_all)
                )
                else None
            )
            hidden_states = layer(
                hidden_states=hidden_states,
                anchor=layer_anchor,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
                anchor_kv=anchor_kv,
                cross_attention_mask=cross_attention_mask,
                cross_position_embeddings=cross_position_embeddings,
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

        # ---- T3-D ADDED: segment embedding (concat_segment mode only) ---------
        # 2 classes (Alternative B grouping):
        #   0 = raw token embedding  (noisy stream)
        #   1 = think-derived hidden state  (anchor + clean streams)
        # The clean stream uses think's last-layer hidden state (not a fresh word
        # embedding lookup), so anchor and clean are the same kind of object. The
        # segment marks "is this an embedding I need to interpret, or a think-processed
        # signal I can use as context?". noisy is the only embedding-derived stream.
        self.is_concat_segment = (config.anchor_injection_mode == "concat_segment")
        if self.is_concat_segment:
            self.segment_embed = nn.Embedding(2, config.resolved_talk_hidden_size)
        else:
            self.segment_embed = None

        # T3-D ADDED: hybrid_xattn flag. In this mode talk's input sequence is L (noisy
        # only); anchor flows in via gated residual at layer 0 AND per-layer cross-attn.
        self.is_hybrid_xattn = (config.anchor_injection_mode == "hybrid_xattn")

        # T3-D ADDED: zero-init delta head (tata-style). When enabled, a Linear with
        # both weight and bias zero-initialised wraps talk_hidden before the anchor skip:
        #     delta_h = delta_head(talk_hidden)        # = 0 at step 0
        #     final   = anchor[:, :L] + delta_h        # = anchor at step 0
        #     logits  = lm_head(final)                  # = LLaDA's logits at step 0 (frozen lm_head)
        # Training learns delta_head's weights to produce useful corrections on top
        # of LLaDA's strong baseline. Pattern: tata/delta_model/models/heads.py.
        if self.is_hybrid_xattn and getattr(config, "use_anchor_delta_head", False):
            self.delta_head = nn.Linear(
                config.resolved_talk_hidden_size,
                config.resolved_talk_hidden_size,
            )
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)
        else:
            self.delta_head = None

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
        if self.segment_embed is not None:
            # Segment embed is part of the talk pathway; gate it on train_talk.
            for p in self.segment_embed.parameters():
                p.requires_grad = self.config.train_talk
        if self.delta_head is not None:
            # Zero-init delta head is part of the talk pathway; gate on train_talk.
            for p in self.delta_head.parameters():
                p.requires_grad = self.config.train_talk

    @torch.no_grad()
    def init_talk_layers_depth_scaled(self) -> None:
        """GPT-NeoX / Megatron-style depth-scaled init for talk transformer layers.

        Vendored recipe from Think-Then-Talk's `manual_init_talk_model`
        (peft_project/T3/Think-Then-Talk/model/modeling_t3.py:L41-L78). Without this,
        from-scratch talk training with uniform std=0.02 init causes the residual stream
        variance to grow with depth, blocking learning -- observed empirically as a
        plateau at loss ~= log(vocab) regardless of LR.

        For each talk decoder layer at index i:
          - Output projections (attention.dense, mlp.down_proj):
                std = initializer_range / sqrt(2 * (i + 1))   (truncated normal)
          - Other projections (attention.query_key_value, mlp.gate_proj, mlp.up_proj):
                std = initializer_range                       (truncated normal)
          - All Linear biases -> zero.

        Norms and embedding-like params are not touched (they keep their constructor
        defaults of ones / Kaiming).

        Should be called once, AFTER VeOmni's `load_model_weights` has run -- otherwise
        VeOmni would overwrite with its uniform-std init. Safe to call multiple times.
        """
        init_std = float(getattr(self.config, "initializer_range", 0.02))

        for layer_idx, block in enumerate(self.talk_model.layers):
            scaled_std = init_std / math.sqrt(2.0 * (layer_idx + 1))
            for name, module in block.named_modules():
                if isinstance(module, nn.Linear):
                    if name.endswith("dense") or name.endswith("down_proj"):
                        # Output projections -- depth-scaled.
                        nn.init.trunc_normal_(
                            module.weight, mean=0.0, std=scaled_std,
                            a=-3 * scaled_std, b=3 * scaled_std,
                        )
                    else:
                        # All other projections -- regular init.
                        nn.init.trunc_normal_(
                            module.weight, mean=0.0, std=init_std,
                            a=-3 * init_std, b=3 * init_std,
                        )
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                # T3-D ADDED: re-init RMSNorm weights to ones inside talk layers.
                # VeOmni's load_model_weights uses trunc_normal(std=init_std) for
                # any unmatched key, including these RMSNorm weights -- which means
                # they end up ~N(0, 0.02^2) instead of the canonical 1.0. That kills
                # downstream magnitudes (RMSNorm output = x * rms_inv * weight); a
                # ~50x reduction means gradients into the delta_head are too small
                # to move it meaningfully in early training.
                if isinstance(module, LLaDA2MoeRMSNorm):
                    nn.init.ones_(module.weight)

        # T3-D ADDED: also re-init the talk_model's final RMSNorm. Same reason as
        # the per-layer RMSNorms above -- VeOmni's loader put it at trunc_normal.
        if hasattr(self.talk_model, "norm"):
            nn.init.ones_(self.talk_model.norm.weight)

        # T3-D ADDED: re-zero the delta_head after VeOmni's load_model_weights step,
        # which initialises unmatched-key params with truncated_normal(std=init_std).
        # Zero-init is load-bearing: it's what guarantees step-0 model output is
        # logit-equivalent to LLaDA exactly.
        if self.delta_head is not None:
            nn.init.zeros_(self.delta_head.weight)
            nn.init.zeros_(self.delta_head.bias)

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
        # T3-D ADDED: extra masks/positions for the concat_segment 3L talk pathway.
        # Training script builds these once at startup and passes via micro_batch.
        attention_mask_3L: Optional[torch.Tensor] = None,
        position_ids_3L: Optional[torch.LongTensor] = None,
        # T3-D ADDED: extra masks/positions for the hybrid_xattn talk pathway.
        attention_mask_L: Optional[torch.Tensor] = None,
        position_ids_L: Optional[torch.LongTensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # 1. Think -- run full LLaDA-2.0-mini, surface all hidden states for the fuser.
        #    Think always processes the 2L sequence; the 2L attention_mask is right.
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
        # 2. Anchor fuser (last_only by default). Shape [B, 2L, D].
        anchor = self.anchor_fuser(think_out.hidden_states)

        # 3. Talk model -- branch on injection mode.
        if self.is_concat_segment:
            # 3a. Build the 3L talk input: [embed(noisy), anchor_for_noisy, embed(clean)]
            #     with segment embedding added.
            assert attention_mask_3L is not None and position_ids_3L is not None, (
                "concat_segment mode requires attention_mask_3L and position_ids_3L kwargs"
            )
            talk_input = self._assemble_talk_input_3L(input_ids, anchor)
            talk_hidden = self.talk_model(
                inputs_embeds=talk_input,
                anchor=None,  # info is in the sequence tokens themselves
                attention_mask=attention_mask_3L,
                position_ids=position_ids_3L,
            )
        elif self.is_hybrid_xattn:
            # 3c. hybrid_xattn mode: talk processes only the noisy half (L tokens).
            #     - Layer 0 residual: anchor[:, :L, :] (think's view of noisy positions).
            #     - Every layer cross-attn: full anchor [B, 2L, D] as K/V.
            assert (
                attention_mask_L is not None
                and position_ids_L is not None
                and cross_attention_mask is not None
                and cross_position_ids is not None
            ), (
                "hybrid_xattn mode requires attention_mask_L, position_ids_L, "
                "cross_attention_mask, and cross_position_ids kwargs"
            )
            two_L = input_ids.shape[1]
            assert two_L % 2 == 0, f"input_ids length {two_L} must be even (2L)"
            L = two_L // 2
            noisy_ids = input_ids[:, :L]
            talk_embeds = self.model.word_embeddings(noisy_ids)
            anchor_noisy = anchor[:, :L, :].contiguous()
            talk_hidden = self.talk_model(
                inputs_embeds=talk_embeds,
                anchor=anchor_noisy,                    # for layer-0 gated residual
                attention_mask=attention_mask_L,
                position_ids=position_ids_L,
                anchor_kv=anchor,                       # full 2L anchor for cross-attn
                cross_attention_mask=cross_attention_mask,
                cross_position_ids=cross_position_ids,
            )
            # T3-D ADDED: residual learning from anchor (tata-style).
            # Two modes (see configuration_think_talk_llada2.py for full docstring):
            #   use_anchor_delta_head=True  -> wrap talk_hidden through zero-init Linear,
            #                                  final = anchor + delta_head(talk_hidden);
            #                                  at step 0, delta = 0 EXACTLY -> logits exactly LLaDA.
            #   add_anchor_skip_residual=True -> simple skip, final = anchor + talk_hidden;
            #                                    at step 0, output ≈ LLaDA + small noise.
            if self.delta_head is not None:
                talk_hidden = anchor_noisy + self.delta_head(talk_hidden)
            elif getattr(self.config, "add_anchor_skip_residual", False):
                talk_hidden = talk_hidden + anchor_noisy
        else:
            # 3b. Gated-residual mode: anchor mixed into talk's residual stream per layer.
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

        # 4. LM head over whatever sequence length talk produced.
        #    In concat_segment mode talk_hidden is [B, 3L, D]; logits[:, :L] is still the
        #    noisy slice the training script wants. In hybrid_xattn mode talk_hidden is
        #    [B, L, D] -- all positions are already the noisy slice.
        logits = self.lm_head(talk_hidden)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    # -------------------------------------------------------------------- helpers

    def _assemble_talk_input_3L(
        self,
        input_ids: torch.LongTensor,
        anchor: torch.Tensor,
    ) -> torch.Tensor:
        """Build talk's 3L input from the 2L input_ids and 2L anchor.

        v5 layout: [embed(noisy), think_hidden_noisy, think_hidden_clean] + segment_embed.
        Only the noisy stream needs a fresh embedding lookup; both anchor and clean come
        from think's last-layer hidden state (anchor here is the anchor_fuser output, which
        in last_only mode is just think_hidden_states[-1]). Avoids redundant re-embedding
        of clean tokens and unifies the representation space of the two context streams.

        Segment scheme (2 classes, Alternative B):
          0 = raw token embedding (noisy)
          1 = think-derived hidden state (anchor + clean)

        Returns: [B, 3L, D]
        """
        assert self.segment_embed is not None, "segment_embed must be initialised"
        assert input_ids is not None, "concat_segment mode requires input_ids"
        B, two_L = input_ids.shape
        assert two_L % 2 == 0, f"input_ids length {two_L} must be even (2L)"
        L = two_L // 2

        noisy_ids = input_ids[:, :L]
        embed_noisy = self.model.word_embeddings(noisy_ids)   # [B, L, D]  (raw token embedding)
        think_hidden_noisy = anchor[:, :L, :]                  # [B, L, D]  (think's hidden at noisy)
        think_hidden_clean = anchor[:, L:, :]                  # [B, L, D]  (think's hidden at clean -- NO re-embed)

        talk_input = torch.cat(
            [embed_noisy, think_hidden_noisy, think_hidden_clean], dim=1,
        )                                                       # [B, 3L, D]

        device = input_ids.device
        segment_ids = torch.cat([
            torch.zeros(L, dtype=torch.long, device=device),    # 0 = raw embedding (noisy)
            torch.ones (L, dtype=torch.long, device=device),    # 1 = think hidden (anchor)
            torch.ones (L, dtype=torch.long, device=device),    # 1 = think hidden (clean)
        ])
        talk_input = talk_input + self.segment_embed(segment_ids)[None]     # [B, 3L, D]
        return talk_input

    @torch.no_grad()
    def run_think_and_anchor(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
        past_key_values=None,
        use_cache: bool = False,
    ):
        """Forward through think + anchor fuser only.

        Training-time call (no cache): returns just the anchor [B, seq_len, D]. Used by
        the talk-only OPUT rollout (brief sec 8.3); seq_len is 2L (the doubled sequence).

        Inference-time call (use_cache=True): extends the think KV cache. Returns
        (anchor_for_new_positions, past_key_values). anchor covers only the query
        positions (length q_len = input_ids.shape[1]); the caller maintains a running
        anchor tensor that accumulates new chunks across blocks. See dInfer/python/dinfer/
        decoding/generate_t3d.py for the block-by-block inference loop.
        """
        think_out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
            output_router_logits=False,
            return_dict=True,
        )
        anchor = self.anchor_fuser(think_out.hidden_states)
        if use_cache:
            return anchor, think_out.past_key_values
        return anchor

    @torch.no_grad()
    def run_talk_block(
        self,
        block_input_ids: Optional[torch.LongTensor] = None,
        *,
        inputs_embeds: Optional[torch.Tensor] = None,
        anchor_so_far: torch.Tensor,
        block_start: int,
        block_end: int,
    ) -> torch.Tensor:
        """Run talk on a single block at inference. Returns logits [B, block_length, V].

        Args:
          block_input_ids: [B, block_length] noisy tokens for the current block.
          inputs_embeds:   [B, block_length, D] pre-built embeddings for the block.
                           Mutually exclusive with block_input_ids. Used by the
                           DMax-style soft-embedding feed in the diagnostic and in
                           generate_t3d's threshold-decoder path.
          anchor_so_far:   [B, block_end, D] anchor accumulated over all positions decoded
                           so far (prompt + committed prior blocks + current block).
          block_start, block_end: absolute positions of the current block in `x`.
        """
        if not self.is_hybrid_xattn:
            raise NotImplementedError(
                "run_talk_block is currently only implemented for hybrid_xattn anchor "
                "injection mode."
            )
        if (block_input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "run_talk_block requires exactly one of block_input_ids or inputs_embeds."
            )

        if inputs_embeds is not None:
            talk_embeds = inputs_embeds
            B, q_len = inputs_embeds.shape[:2]
            device = inputs_embeds.device
        else:
            talk_embeds = self.model.word_embeddings(block_input_ids)
            B, q_len = block_input_ids.shape[:2]
            device = block_input_ids.device
        anchor_block = anchor_so_far[:, block_start:block_end, :].contiguous()

        pos_self = torch.arange(block_start, block_end, dtype=torch.long, device=device)
        pos_self = pos_self.unsqueeze(0).expand(B, -1).contiguous()

        pos_cross_kv = torch.arange(0, block_end, dtype=torch.long, device=device)
        pos_cross_kv = pos_cross_kv.unsqueeze(0).expand(B, -1).contiguous()

        talk_hidden = self.talk_model(
            inputs_embeds=talk_embeds,
            anchor=anchor_block,
            attention_mask=None,
            position_ids=pos_self,
            anchor_kv=anchor_so_far,
            cross_attention_mask=None,
            cross_position_ids=pos_cross_kv,
        )
        if self.delta_head is not None:
            talk_hidden = anchor_block + self.delta_head(talk_hidden)
        elif getattr(self.config, "add_anchor_skip_residual", False):
            talk_hidden = talk_hidden + anchor_block

        return self.lm_head(talk_hidden)

    def run_talk(
        self,
        input_ids: torch.LongTensor,
        anchor: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: torch.LongTensor,
        attention_mask_3L: Optional[torch.Tensor] = None,
        position_ids_3L: Optional[torch.LongTensor] = None,
        attention_mask_L: Optional[torch.Tensor] = None,
        position_ids_L: Optional[torch.LongTensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        cross_position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Forward through talk + LM head only, with a pre-computed anchor. Returns logits.

        Mode-specific args (training script builds the masks/positions once at startup):
          gated_residual: uses attention_mask + position_ids (2L).
          concat_segment: uses attention_mask_3L + position_ids_3L (3L).
          hybrid_xattn:   uses attention_mask_L + position_ids_L (L) for talk self-attn,
                          plus cross_attention_mask + cross_position_ids (2L K/V) for
                          per-layer cross-attn.
        """
        if self.is_concat_segment:
            assert attention_mask_3L is not None and position_ids_3L is not None, (
                "concat_segment mode requires attention_mask_3L and position_ids_3L"
            )
            talk_input = self._assemble_talk_input_3L(input_ids, anchor)
            talk_hidden = self.talk_model(
                inputs_embeds=talk_input,
                anchor=None,
                attention_mask=attention_mask_3L,
                position_ids=position_ids_3L,
            )
        elif self.is_hybrid_xattn:
            assert (
                attention_mask_L is not None
                and position_ids_L is not None
                and cross_attention_mask is not None
                and cross_position_ids is not None
            ), (
                "hybrid_xattn mode requires attention_mask_L, position_ids_L, "
                "cross_attention_mask, and cross_position_ids"
            )
            two_L = input_ids.shape[1]
            assert two_L % 2 == 0, f"input_ids length {two_L} must be even (2L)"
            L = two_L // 2
            noisy_ids = input_ids[:, :L]
            talk_embeds = self.model.word_embeddings(noisy_ids)
            anchor_noisy = anchor[:, :L, :].contiguous()
            talk_hidden = self.talk_model(
                inputs_embeds=talk_embeds,
                anchor=anchor_noisy,
                attention_mask=attention_mask_L,
                position_ids=position_ids_L,
                anchor_kv=anchor,
                cross_attention_mask=cross_attention_mask,
                cross_position_ids=cross_position_ids,
            )
            # T3-D ADDED: anchor delta or simple skip (see ForCausalLM.forward for full doc).
            if self.delta_head is not None:
                talk_hidden = anchor_noisy + self.delta_head(talk_hidden)
            elif getattr(self.config, "add_anchor_skip_residual", False):
                talk_hidden = talk_hidden + anchor_noisy
        else:
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
