# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
"""**DBet** — a lightweight draft model that speeds up diffusion-LLM decoding.

A small trainable drafter rides on a large FROZEN heavy model (LLaDA-2.0-MoE). For a masked block the heavy
runs once; the drafter proposes the masked tokens cheaply and the heavy verifies them. The drafter only
proposes (never commits) and emits a per-position confidence so weak proposals can abstain.

The drafter conditions on the heavy and REFINES its output:
  - soft-conditioning : a soft embedding of the heavy's output distribution, softmax(logits/τ)·W_E
                        (DiffusionGemma-style, added at the input);
  - context injection : the heavy's hidden at a few selected layers, fused and injected as K/V into every
                        drafter layer (DFlash-style; committed context never re-processed);
  - residual decoding : Δh on the heavy's last hidden (zero-init -> starts == heavy), decoded by the frozen
                        heavy lm_head.

Backbone matches LLaDA-2.0 (fused query_key_value, query/key layernorm, dense out, partial rotary, SwiGLU
MLP) so the body can warm-start from the heavy's bottom L layers. Every learned projection is one shared
Gemma-style gated MLP (`DbetGatedMLP`). Train == infer (same forward; committed context injected as KV); the
ONLY training extra is one attention mask making each masked block see exactly its inference-time context.

THE MODEL DOES NOT BUILD MASKS OR COMPUTE LOSS — the caller (data pipeline / training script / inference
loop) owns both. `forward` returns the raw drafter outputs (logits, conf, h_draft, delta); the training
script computes the token CE + asymmetric confidence loss itself. Four masks, each at its proper place
(MASK_DESIGN.md):
  #1 padding + #2 inference-equivalence -> one ready `attention_mask` -> attention;
  #3 loss -> applied by the training script on the returned logits/conf;
  #4 denoise (soft-embed gate) -> `denoise_mask` in HeavyModelConditioning.

Shapes: B=batch, P=#prefix(committed) tokens, C=#canvas(masked) tokens, d=draft hidden, D=heavy hidden,
m=#selected heavy layers, V=vocab, hd=head_dim.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from veomni.utils.import_utils import is_liger_kernel_available

if is_liger_kernel_available():
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction

from .configuration_dbet import DbetConfig
from ..llada2_moe.modeling_llada2_moe import (
    LLaDA2MoeRMSNorm,
    LLaDA2MoeRotaryEmbedding,
    LLaDA2MoePreTrainedModel,
    LLaDA2MoeModelLM,
    rotate_half,
    repeat_kv,
)


def _apply_rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
    """Apply rotary embedding to ONE tensor (q or k), supporting partial rotary. Mirrors LLaDA-2.0's
    `apply_rotary_pos_emb` but for a single tensor, so q (canvas) and k (prefix+canvas) — which have
    different lengths here — can be rotated with their own cos/sin slices.
    x [B,h,T,hd]; cos/sin [B,T,rope_dim] -> [B,h,T,hd]."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return torch.cat([x_embed, x_pass], dim=-1)


# NOTE: the DMax-native heavy decode (decode_uniform: grid-aligned blocks, left-to-right threshold commit,
# soft-embedding reveal) is DEFERRED to the inference stage. To match DMax/LLaDA-2.0's framework it will live
# in a separate inference wrapper (a `DbetDiffusionLLM`, mirroring dInfer's DiffusionLLM classes), NOT in the
# model. `heavy_forward` (pure passthrough) stays on the model; `heavy_generate` was removed for now.


# ======================================================================================
# Shared building block — one gated MLP behind every learned projection
# ======================================================================================
class DbetGatedMLP(nn.Module):
    """Gemma-style gated SwiGLU feed-forward (DiffusionGemmaSelfConditioning, modular_diffusion_gemma.py:608).
        y = down_proj( act(gate_proj(pre_norm(x))) * up_proj(pre_norm(x)) )
        y = y + residual_to        # optional (shapes must match)
        y = no_scale_rms(y)        # optional post_norm (DiffusionGemma uses a no-scale RMSNorm)
    `zero_init_out` zero-inits down_proj so the module outputs 0 at init (the Δh head's "start == heavy").
    gate/up/down are named to match LLaDA2MoeMLP so the body MLP can warm-start from the heavy.
    """

    def __init__(self, d_in, d_intermediate, d_out, *, act="silu", pre_norm=True, post_norm=False, zero_init_out=False, eps=1e-6):
        super().__init__()
        # Reuse the heavy's RMSNorm (== DiffusionGemma's RMSNorm up to rsqrt-vs-pow). DiffusionGemma's
        # post_norm is with_scale=False; the repo has no no-scale variant, so we use the scaled RMSNorm here
        # too (a harmless extra d-length scale, init 1).
        self.pre_norm = LLaDA2MoeRMSNorm(d_in, eps=eps) if pre_norm else None
        self.gate_proj = nn.Linear(d_in, d_intermediate, bias=False)
        self.up_proj = nn.Linear(d_in, d_intermediate, bias=False)
        self.down_proj = nn.Linear(d_intermediate, d_out, bias=False)
        self.act_fn = ACT2FN[act]                                     # configurable (config.draft_hidden_act), like LLaDA-2.0
        self.use_liger = act == "silu" and is_liger_kernel_available()  # fused SiLU·mul fast path (silu only)
        self.post_norm = LLaDA2MoeRMSNorm(d_out, eps=eps) if post_norm else None
        if zero_init_out:
            nn.init.zeros_(self.down_proj.weight)

    def forward(self, x: torch.Tensor, residual_to: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.pre_norm(x) if self.pre_norm is not None else x
        if self.use_liger:
            y = self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(h), self.up_proj(h)))
        else:
            y = self.down_proj(self.act_fn(self.gate_proj(h)) * self.up_proj(h))
        if residual_to is not None:
            y = y + residual_to
        if self.post_norm is not None:
            y = self.post_norm(y)
        return y


# ======================================================================================
# Conditioning — turn the heavy's signals into the drafter's input
# ======================================================================================
class SoftEmbed(nn.Module):
    """Soft-embed: S = DbetGatedMLP( softmax(logits/τ) · W_E ). Full probability-weighted embedding of the
    heavy's logits (not a top-K blur), through the shared gated MLP. W_E is the heavy's frozen embedding."""

    def __init__(self, config: DbetConfig):
        super().__init__()
        self.mlp = DbetGatedMLP(
            config.hidden_size, config.resolved_fuse_hidden_size, config.resolved_draft_hidden_size,
            pre_norm=True, post_norm=False, act=config.draft_hidden_act, eps=config.rms_norm_eps,
        )

    def forward(self, heavy_logits: torch.Tensor, embed_weight: torch.Tensor, tau: float) -> torch.Tensor:
        probs = torch.softmax(heavy_logits.float() / tau, dim=-1).to(embed_weight.dtype)  # [B,C,V]
        soft = probs @ embed_weight                                                       # [B,C,D]
        return self.mlp(soft)                                                             # [B,C,d]


class HiddenFuse(nn.Module):
    """Fuse the heavy's selected-layer hidden into one draft-width feature via the shared gated MLP
    (m·D -> d). Used as the denoise fuse (-> F_dn, added at input) and as the prefix fuse(s) (-> F_pre^l,
    injected as K/V). Reads only context / already-committed hidden -> no leakage."""

    def __init__(self, config: DbetConfig):
        super().__init__()
        self.mlp = DbetGatedMLP(
            config.m * config.hidden_size, config.resolved_fuse_hidden_size, config.resolved_draft_hidden_size,
            pre_norm=True, post_norm=True, act=config.draft_hidden_act, eps=config.rms_norm_eps,
        )

    def forward(self, h_sel: torch.Tensor) -> torch.Tensor:
        return self.mlp(h_sel)


class PrefixFuse(nn.Module):
    """Fuse committed-context heavy hidden into the per-layer KV-injection features (F_pre^l).

    ONE shared trunk (m·D -> inter) with a single output head producing all layers at once:
      - per_layer_prefix_fuse=True  : head -> L·d, split into L features of width d (one per draft layer).
      - per_layer_prefix_fuse=False : head -> d, the same feature reused for every layer (DFlash's shared F).
    This is ~Lx cheaper than L independent fuses (the expensive m·D input projection is shared once; only the
    output head scales with L). Per-layer LINEAR freedom is still preserved because each draft layer
    re-projects its feature through its own query_key_value (-> key_layernorm) in attention; so no post_norm
    here. No leakage: reads only the committed-context hidden of the last heavy run."""

    def __init__(self, config: DbetConfig):
        super().__init__()
        self.per_layer = config.per_layer_prefix_fuse
        self.num_layers = config.draft_num_layers
        self.d = config.resolved_draft_hidden_size
        out = self.num_layers * self.d if self.per_layer else self.d
        self.mlp = DbetGatedMLP(
            config.m * config.hidden_size, config.resolved_fuse_hidden_size, out,
            pre_norm=True, post_norm=False, act=config.draft_hidden_act, eps=config.rms_norm_eps,
        )

    def forward(self, h_sel_prefix: torch.Tensor):
        """h_sel_prefix [B,P,m*D] -> tuple of L features, each [B,P,d] (one per draft layer)."""
        f = self.mlp(h_sel_prefix)
        if self.per_layer:
            b, p, _ = f.shape
            return f.view(b, p, self.num_layers, self.d).unbind(2)   # L × [B,P,d]
        return (f,) * self.num_layers                                # shared feature, reused (no copy)


class HeavyModelConditioning(nn.Module):
    """Input assembly: x = post_norm( E(input_ids) + denoise_mask*SoftEmbed + DenoiseFuse(H_sel[denoise]) ).
    Generalizes DiffusionGemmaSelfConditioning by adding the fused heavy-hidden term. Uses the heavy's frozen
    embedding for E(input_ids); an input projection handles the d != D case (a thinner drafter)."""

    def __init__(self, config: DbetConfig):
        super().__init__()
        self.soft_embed = SoftEmbed(config)
        self.denoise_fuse = HiddenFuse(config)
        d, D = config.resolved_draft_hidden_size, config.hidden_size
        self.input_proj = nn.Linear(D, d, bias=False) if d != D else None
        self.post_norm = LLaDA2MoeRMSNorm(d, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        heavy_logits: torch.Tensor,
        h_sel_denoise: torch.Tensor,
        frozen_embed: nn.Embedding,
        tau: float,
        denoise_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        e = frozen_embed(input_ids)                                   # [B,C,D]
        if self.input_proj is not None:
            e = self.input_proj(e)                                    # [B,C,d]
        s = self.soft_embed(heavy_logits, frozen_embed.weight, tau)   # [B,C,d]  (MASK #4 gates this branch)
        if denoise_mask is not None:
            mask = denoise_mask.to(s.dtype)
            mask = mask[:, None, None] if mask.dim() == 1 else mask[..., None]
            s = s * mask
        f_dn = self.denoise_fuse(h_sel_denoise)                       # [B,C,d]
        return self.post_norm(e + s + f_dn)


# ======================================================================================
# Backbone — attention (DFlash KV injection + prefix cache) + decoder layer
# ======================================================================================
class DbetAttention(nn.Module):
    """Bidirectional attention with DFlash-style KV injection of the committed context. Backbone matches
    LLaDA-2.0 (fused query_key_value, query/key layernorm, dense out, partial rotary) so it warm-starts from
    the heavy. Queries = the C canvas tokens; keys/values = [injected prefix ; canvas]. The committed prefix
    is constant across the draft rounds between two heavy passes, so its (rope'd) K/V is cached and reused.
    `attention_mask` is a READY mask from the caller — the model never builds masks (MASK_DESIGN.md).

    Pluggable attention backend (config._attn_implementation), same selection as LLaDA-2.0, so it runs fast
    on H200:
      - "sdpa" (default): F.scaled_dot_product_attention -> dispatches to FlashAttention-2 / mem-efficient
        kernels automatically; `attention_mask` is a 4D additive/bool mask (or None for the unmasked fast path).
      - "flex_attention": torch FlexAttention; `attention_mask` is a `BlockMask` built by the caller — the
        efficient way to apply the structured block-diffusion mask (#2) at training (matches LLaDA-2.0/DMax).
      - "eager": plain softmax matmul (debug / output_attentions).
    (True varlen "flash_attention_2" needs unpadded/packed inputs with no arbitrary mask — out of scope here;
    use packing + "sdpa", which already lands on the flash kernel.)
    """

    def __init__(self, config: DbetConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_impl = getattr(config, "_attn_implementation", "sdpa")
        self.hidden_size = config.resolved_draft_hidden_size
        self.num_heads = config.resolved_draft_num_attention_heads
        self.num_kv_heads = config.resolved_draft_num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.draft_head_dim
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.query_key_value = nn.Linear(
            self.hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim, bias=config.use_qkv_bias,
        )
        self.query_layernorm = LLaDA2MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = LLaDA2MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.use_bias)

    def _run_attention(self, q, k, v, attention_mask):
        """Dispatch to the configured efficient backend. q [B,H,C,hd]; k,v [B,H,P+C,hd] (already GQA-expanded);
        attention_mask: 4D additive/bool for sdpa/eager, or a flex BlockMask. -> [B,H,C,hd]."""
        dropout = self.attention_dropout if self.training else 0.0
        if self.attn_impl == "flex_attention":
            from torch.nn.attention.flex_attention import flex_attention
            return flex_attention(q, k, v, block_mask=attention_mask, scale=self.scaling)
        if self.attn_impl == "eager":
            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
            if attention_mask is not None:
                scores = scores + attention_mask                              # additive mask
            attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            attn = F.dropout(attn, p=dropout, training=self.training)
            return torch.matmul(attn, v)
        return F.scaled_dot_product_attention(                                # "sdpa" (FlashAttention on H200)
            q, k, v, attn_mask=attention_mask, dropout_p=dropout, is_causal=False, scale=self.scaling,
        )

    def _project(self, x: torch.Tensor):
        """x [B,T,d] -> (q [B,H,T,hd], k [B,Hkv,T,hd], v [B,Hkv,T,hd]) via the fused projection."""
        b, t, _ = x.shape
        qkv = self.query_key_value(x).view(b, t, self.num_heads + 2 * self.num_kv_heads, self.head_dim)
        q, k, v = qkv.split([self.num_heads, self.num_kv_heads, self.num_kv_heads], dim=2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prefix_kv: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, c, _ = hidden_states.shape
        q, k_can, v_can = self._project(hidden_states)
        q, k_can = self.query_layernorm(q), self.key_layernorm(k_can)
        cos_can, sin_can = cos[:, -c:], sin[:, -c:]                       # canvas = last C positions
        q = _apply_rope_single(q, cos_can, sin_can)
        k_can = _apply_rope_single(k_can, cos_can, sin_can)

        # --- committed-context K/V: build+cache on the first round, else read the cache (DFlash + Gemma cache) ---
        if prefix_kv is not None:
            p = prefix_kv.shape[1]
            _, k_pre, v_pre = self._project(prefix_kv)
            k_pre = self.key_layernorm(k_pre)
            k_pre = _apply_rope_single(k_pre, cos[:, :p], sin[:, :p])     # prefix keeps its own positions
            if past_key_values is not None:
                k_pre, v_pre = past_key_values.update(k_pre, v_pre, self.layer_idx)
        elif past_key_values is not None and len(past_key_values.key_cache) > self.layer_idx:
            k_pre, v_pre = past_key_values.key_cache[self.layer_idx], past_key_values.value_cache[self.layer_idx]
        else:
            k_pre = v_pre = None

        if k_pre is not None:
            k = torch.cat([k_pre, k_can], dim=2)
            v = torch.cat([v_pre, v_can], dim=2)
        else:
            k, v = k_can, v_can
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        attn = self._run_attention(q, k, v, attention_mask)              # sdpa (flash) | flex_attention | eager
        attn = attn.transpose(1, 2).reshape(b, c, -1)
        return self.dense(attn)


class DbetDecoderLayer(nn.Module):
    """One draft layer (LLaDA-2.0 pre-norm block): residual + DbetAttention, residual + SwiGLU MLP. The MLP
    is a DbetGatedMLP (gate/up/down named like LLaDA2MoeMLP) so the whole layer warm-starts from the heavy."""

    def __init__(self, config: DbetConfig, layer_idx: int):
        super().__init__()
        d = config.resolved_draft_hidden_size
        self.attention = DbetAttention(config, layer_idx)
        self.mlp = DbetGatedMLP(d, config.resolved_draft_intermediate_size, d, pre_norm=False, post_norm=False, act=config.draft_hidden_act, eps=config.rms_norm_eps)
        self.input_layernorm = LLaDA2MoeRMSNorm(d, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LLaDA2MoeRMSNorm(d, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prefix_kv: Optional[torch.Tensor],
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.attention(h, prefix_kv, cos, sin, attention_mask, past_key_values, cache_position)
        hidden_states = residual + h
        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)
        h = self.mlp(h)
        return residual + h


def _make_decoder_layer(config: DbetConfig, layer_idx: int) -> nn.Module:
    """Layer factory on `config.draft_layer_type`: "dense" -> DbetDecoderLayer; "moe" -> TODO."""
    if config.draft_layer_type == "dense":
        return DbetDecoderLayer(config, layer_idx)
    raise ValueError(
        f"draft_layer_type={config.draft_layer_type!r} not supported yet "
        f"(\"moe\" = a LLaDA2MoeDecoderLayer-style sparse block is a TODO)."
    )


# ======================================================================================
# Heads — Δh (frozen-head decode) + confidence
# ======================================================================================
class DbetDeltaHead(nn.Module):
    """h_draft = h_last + Δh(hb). DbetGatedMLP (draft_hidden -> heavy_hidden) with ZERO-INIT output, so Δh=0
    at step 0 -> logits == the heavy's exactly; training learns only the residual."""

    def __init__(self, config: DbetConfig):
        super().__init__()
        self.mlp = DbetGatedMLP(
            config.resolved_draft_hidden_size, config.resolved_head_intermediate_size, config.hidden_size,
            pre_norm=True, post_norm=False, zero_init_out=True, act=config.draft_hidden_act, eps=config.rms_norm_eps,
        )

    def forward(self, hb: torch.Tensor, h_last: torch.Tensor) -> torch.Tensor:
        return h_last + self.mlp(hb)


class DbetConfidenceHead(nn.Module):
    """c = σ( DbetGatedMLP(hb) ) ∈ [0,1] — "will the heavy accept this draft?". An MLP (draft_hidden -> 1)."""

    def __init__(self, config: DbetConfig):
        super().__init__()
        self.mlp = DbetGatedMLP(
            config.resolved_draft_hidden_size, config.resolved_head_intermediate_size, 1,
            pre_norm=True, post_norm=False, act=config.draft_hidden_act, eps=config.rms_norm_eps,
        )

    def forward(self, hb: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(hb)).squeeze(-1)


# ======================================================================================
# Drafter stack — the trainable body (everything except the frozen heavy)
# ======================================================================================
class DbetDraftStack(nn.Module):
    """Structured like a DiffusionGemma decoder: assemble the input, embed positions once, run the L layers
    (each reading the injected/cached prefix KV under the caller's `attention_mask`), final-norm, decode.
    Owns the conditioning, the prefix fuses, the L layers, the final norm, the Δh + confidence heads, and
    FROZEN references to the heavy's embedding / lm_head / final-norm."""

    def __init__(self, config: DbetConfig, frozen_embed: nn.Embedding, frozen_lm_head: nn.Linear, frozen_final_norm: nn.Module):
        super().__init__()
        self.config = config
        self.frozen_embed = frozen_embed
        self.frozen_lm_head = frozen_lm_head
        self.frozen_final_norm = frozen_final_norm
        self.conditioning = HeavyModelConditioning(config)
        self.prefix_fuse = PrefixFuse(config)   # one shared trunk -> L per-layer features (or 1 shared)
        self.rotary_emb = LLaDA2MoeRotaryEmbedding(config)
        self.layers = nn.ModuleList([_make_decoder_layer(config, i) for i in range(config.draft_num_layers)])
        self.norm = LLaDA2MoeRMSNorm(config.resolved_draft_hidden_size, eps=config.rms_norm_eps)
        self.delta_head = DbetDeltaHead(config)
        self.conf_head = DbetConfidenceHead(config) if config.use_confidence_head else None

    def forward(
        self,
        input_ids: torch.Tensor,
        heavy_logits: torch.Tensor,
        h_sel_denoise: torch.Tensor,
        h_last_denoise: torch.Tensor,
        h_sel_prefix: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        denoise_mask: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
    ) -> dict:
        cfg = self.config
        tau = tau if tau is not None else cfg.soft_embed_temp
        x = self.conditioning(input_ids, heavy_logits, h_sel_denoise, self.frozen_embed, tau, denoise_mask)
        b, c, _ = x.shape

        # committed-context length P (for positions): from the prefix feature, else the cache, else 0.
        if h_sel_prefix is not None:
            p = h_sel_prefix.shape[1]
        elif past_key_values is not None and len(past_key_values.key_cache) > 0:
            p = past_key_values.key_cache[0].shape[2]
        else:
            p = 0

        if position_ids is None:
            position_ids = torch.arange(p + c, device=x.device).unsqueeze(0).expand(b, -1)
        cos, sin = self.rotary_emb(x, position_ids)                       # [B, P+C, rope_dim], computed ONCE

        f_pre = self.prefix_fuse(h_sel_prefix) if h_sel_prefix is not None else None  # tuple of L × [B,P,d] | None
        for i, layer in enumerate(self.layers):
            pk = None if f_pre is None else f_pre[i]
            x = layer(x, pk, cos, sin, attention_mask, past_key_values)

        hb = self.norm(x)
        h_draft = self.delta_head(hb, h_last_denoise)
        logits = self.frozen_lm_head(self.frozen_final_norm(h_draft))
        conf = self.conf_head(hb) if self.conf_head is not None else None
        return {"logits": logits, "conf": conf, "h_draft": h_draft, "delta": h_draft - h_last_denoise}


# ======================================================================================
# Top-level — frozen heavy + drafter
# ======================================================================================
class DbetForDraftDecoding(LLaDA2MoePreTrainedModel):
    """End-to-end DBet: a FROZEN heavy (LLaDA2MoeModelLM = DMax) + the trainable `DbetDraftStack`. The
    drafter is a proposer the heavy verifies. `extract_heavy_signals` runs the heavy once; `draft_forward`
    runs one draft forward; `forward` chains them for training/eval. All masks come from the caller."""

    config_class = DbetConfig
    _no_split_modules = ["DbetDecoderLayer"]

    def __init__(self, config: DbetConfig):
        super().__init__(config)
        self.heavy = LLaDA2MoeModelLM(config)
        embed = self.heavy.get_input_embeddings()
        lm_head = self.heavy.get_output_embeddings()
        final_norm = self.heavy.model.norm
        self.draft = DbetDraftStack(config, embed, lm_head, final_norm)
        self._apply_freeze_flags()
        if config.warmstart_from_heavy_bottom:
            self.init_draft_layers_warmstart()

    # ---- embeddings (delegate to the frozen heavy) ----
    def get_input_embeddings(self) -> nn.Module:
        return self.heavy.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.heavy.set_input_embeddings(value)

    # ---- heavy-only passthrough (reproduce DMax exactly, NO drafter) ----
    def heavy_forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, position_ids=None,
                      output_hidden_states=False):
        """One FROZEN-heavy forward (the DMax backbone), bypassing the drafter entirely. Thin wrapper so an
        external decode loop can drive the heavy through this model. -> the heavy's CausalLM output
        (`.logits` [B,N,V], `.hidden_states` if requested)."""
        return self.heavy(
            input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, use_cache=False, output_hidden_states=output_hidden_states,
            output_router_logits=False, return_dict=True,
        )

    # ---- freezing / warm-start ----
    def _apply_freeze_flags(self) -> None:
        if not self.config.train_heavy:
            for p in self.heavy.parameters():
                p.requires_grad_(False)
        # The reused heavy pieces are held by the stack too; freeze flags are about those refs.
        if self.config.freeze_embedding:
            for p in self.draft.frozen_embed.parameters():
                p.requires_grad_(False)
        if self.config.freeze_lm_head:
            for p in self.draft.frozen_lm_head.parameters():
                p.requires_grad_(False)
        if self.config.freeze_final_norm:
            for p in self.draft.frozen_final_norm.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def init_draft_layers_warmstart(self) -> None:
        """Copy the heavy's bottom-L decoder-layer weights into the L draft layers (shape-permitting), then
        re-zero the Δh head. No-op if disabled. NOTE: only valid when a heavy bottom layer is DENSE and the
        draft widths match the heavy (the default). MoE heavy layers / thinner drafts need separate handling
        (open decision in the design doc)."""
        if not self.config.warmstart_from_heavy_bottom:
            return
        heavy_layers = self.heavy.model.layers
        for i, draft_layer in enumerate(self.draft.layers):
            if i >= len(heavy_layers):
                break
            src = heavy_layers[i].state_dict()
            missing = draft_layer.load_state_dict(src, strict=False)  # dense subset copies; MoE/extra keys skipped
            _ = missing
        # keep the Δh residual at zero after copying
        nn.init.zeros_(self.draft.delta_head.mlp.down_proj.weight)

    # ---- heavy pass + signal extraction ----
    def _split_prefix_denoise(self, input_ids: torch.Tensor):
        """Boolean masks (prefix = committed, canvas = mask_token_id). input_ids [B,N] -> (prefix, canvas)."""
        canvas = input_ids == self.config.mask_token_id
        return ~canvas, canvas

    @torch.no_grad()
    def extract_heavy_signals(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> dict:
        """Run the FROZEN heavy once; capture hidden at config.sel_layers_list (concat -> m*D), the last-layer
        hidden, and the logits; split prefix/canvas. Returns the dict consumed by `draft_forward`. The heavy
        is bidirectional within a block, so a single pass over [prefix ; canvas] gives all signals."""
        out = self.heavy(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, use_cache=False, return_dict=True,
        )
        hs = out.hidden_states                                  # tuple of [B,N,D], len = num_layers+1
        sel = torch.cat([hs[i] for i in self.config.sel_layers_list], dim=-1)  # [B,N,m*D]
        h_last = hs[-1]                                         # [B,N,D]
        logits = out.logits                                    # [B,N,V]
        prefix_idx, canvas_idx = self._split_prefix_denoise(input_ids)
        return {
            "input_ids": input_ids, "logits": logits, "h_sel": sel, "h_last": h_last,
            "prefix_idx": prefix_idx, "canvas_idx": canvas_idx,
        }

    # ---- one draft forward ----
    def draft_forward(
        self,
        signals: dict,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        denoise_mask: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
    ) -> dict:
        """Slice the heavy signals into prefix/canvas and run `DbetDraftStack.forward`. Assumes a contiguous
        [prefix ; canvas] layout per sequence (the common single-block case); for ragged layouts the caller
        should pre-slice and pass tensors directly. -> dict(logits, conf, h_draft, delta)."""
        ids = signals["input_ids"]
        prefix_idx, canvas_idx = signals["prefix_idx"], signals["canvas_idx"]
        b = ids.shape[0]

        def gather(t, idx):
            # [B,N,*] -> [B,K,*] selecting idx (assumes equal K per row; the single-block training case).
            k = int(idx[0].sum())
            return t[idx].view(b, k, *t.shape[2:])

        input_ids = ids[canvas_idx].view(b, int(canvas_idx[0].sum()))
        return self.draft(
            input_ids=input_ids,
            heavy_logits=gather(signals["logits"], canvas_idx),
            h_sel_denoise=gather(signals["h_sel"], canvas_idx),
            h_last_denoise=gather(signals["h_last"], canvas_idx),
            h_sel_prefix=gather(signals["h_sel"], prefix_idx) if prefix_idx.any() else None,
            past_key_values=past_key_values, attention_mask=attention_mask,
            position_ids=position_ids, denoise_mask=denoise_mask, tau=tau,
        )

    # ---- end-to-end (training / eval) ----
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        denoise_mask: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
    ) -> dict:
        """extract_heavy_signals -> draft_forward; returns the raw drafter outputs only. The model does NOT
        compute loss — the training script owns that (it has the labels, the loss/accept masks and the loss
        weighting). `attention_mask` (#1[+#2]) and `denoise_mask` (#4) are passed through.
        Returns dict: logits [B,C,V], conf [B,C] (or None), h_draft [B,C,D], delta [B,C,D]. Everything a
        training script needs to compute token CE + the asymmetric confidence loss itself."""
        signals = self.extract_heavy_signals(input_ids, attention_mask)
        return self.draft_forward(
            signals, attention_mask=attention_mask, position_ids=position_ids,
            denoise_mask=denoise_mask, tau=tau,
        )
