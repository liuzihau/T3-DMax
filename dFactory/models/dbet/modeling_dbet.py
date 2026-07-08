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
try:
    from veomni.utils.import_utils import is_liger_kernel_available
except ImportError:   # veomni absent (SGLang inference env) -> no liger fast path (functionally identical)
    def is_liger_kernel_available():
        return False

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


def _cache_seq_len(cache) -> int:
    """Seq-dim length of a (possibly empty) Cache, robust across transformers versions."""
    try:
        return int(cache.get_seq_length())
    except Exception:
        return 0


def _cache_layer_kv(cache, layer_idx):
    """Read (keys, values) for one layer of a DynamicCache, robust to the transformers API change that
    replaced the `key_cache`/`value_cache` lists with a `layers` list of DynamicLayer(.keys/.values)."""
    if hasattr(cache, "key_cache"):                              # transformers < ~4.54
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    layer = cache.layers[layer_idx]                              # transformers >= ~4.54
    k = getattr(layer, "keys", None)
    v = getattr(layer, "values", None)
    if k is None:                                               # very defensive: alternate attr names
        k, v = getattr(layer, "key_cache", None), getattr(layer, "value_cache", None)
    return k, v


class FixedPrefixCache:
    """FIXED-size cross-block prefix KV so the draft forward keeps a STATIC shape -> torch.compile stays on its
    fast path (a growing DynamicCache makes every block a new shape -> recompile/dynamic -> ~eager speed). Buffers
    are [n_layers] x [1, kv_heads, max_len, head_dim]; the attention always reads the full max_len buffer and the
    caller passes a bool mask hiding the unfilled [settled:max_len]. rope is applied at WRITE time (as in the
    DynamicCache path), so the pad rows (masked) never matter."""

    def __init__(self, n_layers, kv_heads, head_dim, max_len, dtype, device):
        self.k = [torch.zeros(1, kv_heads, max_len, head_dim, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.zeros(1, kv_heads, max_len, head_dim, dtype=dtype, device=device) for _ in range(n_layers)]
        self.max_len = int(max_len)
        self.settled = 0

    def reset(self):
        self.settled = 0
        for t in self.k:
            t.zero_()
        for t in self.v:
            t.zero_()

    def write(self, layer_idx, k, v):                           # k,v [1,H,n,hd] -> buffer[settled:settled+n]
        e = min(self.settled + k.shape[2], self.max_len)        # clamp (drop far prefix if gen exceeds max_len)
        n = e - self.settled
        if n <= 0:
            return
        self.k[layer_idx][:, :, self.settled:e] = k[:, :, :n]
        self.v[layer_idx][:, :, self.settled:e] = v[:, :, :n]


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
        # per_layer_denoise_fuse: the canvas h_sel is injected PER-LAYER (in DbetDraftStack), not fused here at input
        self.denoise_fuse = None if config.per_layer_denoise_fuse else HiddenFuse(config)
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
        if self.denoise_fuse is not None:                            # old path: fuse the canvas h_sel at the input
            return self.post_norm(e + s + self.denoise_fuse(h_sel_denoise))
        return self.post_norm(e + s)                                 # per-layer path: denoise injected in the stack


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
        elif isinstance(past_key_values, FixedPrefixCache):
            k_pre = past_key_values.k[self.layer_idx]            # full max_len buffer (STATIC shape; mask hides pad)
            v_pre = past_key_values.v[self.layer_idx]
        elif past_key_values is not None and _cache_seq_len(past_key_values) > 0:
            k_pre, v_pre = _cache_layer_kv(past_key_values, self.layer_idx)
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

    @torch.no_grad()
    def cache_prefix(self, prefix_feat, cos, sin, past_key_values):
        """Incrementally APPEND a settled-prefix chunk's K/V to the cross-block prefix cache. `prefix_feat`
        [B,n,d] = the PrefixFuse feature for the chunk; `cos`/`sin` [B,n,rope] its rope at its TRUE absolute
        positions. Mirrors the `prefix_kv` branch of `forward` (project -> key-LN -> rope k), but appends a chunk
        at arbitrary positions so blocks are added one at a time as they settle. v is stored raw (as in forward)."""
        _, k_pre, v_pre = self._project(prefix_feat)
        k_pre = self.key_layernorm(k_pre)
        k_pre = _apply_rope_single(k_pre, cos, sin)
        if isinstance(past_key_values, FixedPrefixCache):
            past_key_values.write(self.layer_idx, k_pre, v_pre)  # write at settled offset (fixed buffer)
        else:
            past_key_values.update(k_pre, v_pre, self.layer_idx)


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
        # Hold the heavy's embed/lm_head/final-norm as PLAIN attributes (object.__setattr__ bypasses
        # nn.Module registration), so they are NOT duplicated in the drafter's state_dict — they already
        # live under `heavy.*`. Avoids the shared-tensor error in save_pretrained(safetensors) and a second
        # copy of those weights. Runtime access (self.frozen_embed(...)) and .parameters() still work; the
        # heavy owns their device/dtype.
        object.__setattr__(self, "frozen_embed", frozen_embed)
        object.__setattr__(self, "frozen_lm_head", frozen_lm_head)
        object.__setattr__(self, "frozen_final_norm", frozen_final_norm)
        self.conditioning = HeavyModelConditioning(config)
        self.prefix_fuse = PrefixFuse(config)   # one shared trunk -> L per-layer features (or 1 shared)
        # per_layer_denoise_fuse: fuse the canvas h_sel S->L*d (PrefixFuse-style) + inject per layer via
        # RMSNorm(x + f_dn[l]) (same combine as the input conditioning, applied at every layer). New params.
        if config.per_layer_denoise_fuse:
            self.denoise_layer_fuse = PrefixFuse(config)             # -> tuple of L x [B,C,d]
            self.denoise_combine_norms = nn.ModuleList(
                [LLaDA2MoeRMSNorm(config.resolved_draft_hidden_size, eps=config.rms_norm_eps)
                 for _ in range(config.draft_num_layers)])
        else:
            self.denoise_layer_fuse = None
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

        # Positions: the decode passes a FIXED-shape position_ids so torch.compile never reads the Python int
        # settled/seq-len (which would guard -> recompile per block). Only build defaults when not supplied.
        if position_ids is None:
            if h_sel_prefix is not None:
                p = h_sel_prefix.shape[1]
            elif isinstance(past_key_values, FixedPrefixCache):
                p = past_key_values.settled                      # logical prefix len (buffer is max_len; mask hides pad)
            elif past_key_values is not None and _cache_seq_len(past_key_values) > 0:
                p = _cache_seq_len(past_key_values)
            else:
                p = 0
            position_ids = torch.arange(p + c, device=x.device).unsqueeze(0).expand(b, -1)
        cos, sin = self.rotary_emb(x, position_ids)                       # [B, P+C, rope_dim], computed ONCE

        f_pre = self.prefix_fuse(h_sel_prefix) if h_sel_prefix is not None else None  # tuple of L × [B,P,d] | None
        f_dn = self.denoise_layer_fuse(h_sel_denoise) if self.denoise_layer_fuse is not None else None  # L × [B,C,d]
        for i, layer in enumerate(self.layers):
            if f_dn is not None:                                     # per-layer canvas h_sel: RMSNorm(x + f_dn[l])
                x = self.denoise_combine_norms[i](x + f_dn[i])
            pk = None if f_pre is None else f_pre[i]
            x = layer(x, pk, cos, sin, attention_mask, past_key_values)

        hb = self.norm(x)
        h_draft = self.delta_head(hb, h_last_denoise)
        logits = self.frozen_lm_head(self.frozen_final_norm(h_draft))
        conf = self.conf_head(hb) if self.conf_head is not None else None
        return {"logits": logits, "conf": conf, "h_draft": h_draft, "delta": h_draft - h_last_denoise}

    @torch.no_grad()
    def extend_prefix_cache(self, h_sel_delta, position_ids, past_key_values):
        """Cross-block prefix-KV cache: fuse a newly-SETTLED prefix chunk (its `h_sel` at absolute
        `position_ids`) and APPEND its per-layer K/V to `past_key_values`. PrefixFuse + the per-layer projection /
        key-LN / rope are all POSITION-WISE, so fusing the chunk == fusing the whole prefix and slicing -> the
        incremental cache is exact (bit-identical, no non-associativity). Call once per block with the block that
        just settled, BEFORE that block's draft rounds. h_sel_delta [B,n,m*D]; position_ids [B,n]."""
        f_pre = self.prefix_fuse(h_sel_delta)                        # tuple L x [B,n,d] (or shared feature)
        cos, sin = self.rotary_emb(h_sel_delta, position_ids)        # rope at the chunk's TRUE absolute positions
        for i, layer in enumerate(self.layers):
            layer.attention.cache_prefix(f_pre[i], cos, sin, past_key_values)
        if isinstance(past_key_values, FixedPrefixCache):            # all layers wrote at the same offset; advance once
            past_key_values.settled = min(past_key_values.settled + h_sel_delta.shape[1], past_key_values.max_len)

    def new_fixed_prefix_cache(self, max_len, device, dtype=torch.bfloat16):
        """A FixedPrefixCache sized for THIS drafter (static-shape prefix -> torch.compile fast path in the decode)."""
        cfg = self.config
        hd = cfg.resolved_draft_hidden_size // cfg.draft_num_attention_heads
        return FixedPrefixCache(cfg.draft_num_layers, cfg.draft_num_key_value_heads, hd, max_len, dtype, device)


# ======================================================================================
# Top-level — frozen heavy + drafter
# ======================================================================================
class DbetForDraftDecoding(LLaDA2MoePreTrainedModel):
    """End-to-end DBet: a FROZEN heavy (LLaDA2MoeModelLM = DMax) + the trainable `DbetDraftStack`. The
    drafter is a proposer the heavy verifies. `extract_heavy_signals` runs the heavy once; `draft_forward`
    runs one draft forward; `forward` chains them for training/eval. All masks come from the caller."""

    config_class = DbetConfig
    _no_split_modules = ["DbetDecoderLayer", "LLaDA2MoeDecoderLayer"]   # shard BOTH draft + heavy layers per-layer (FSDP)

    def __init__(self, config: DbetConfig, _heavy: Optional[nn.Module] = None):
        super().__init__(config)
        # `_heavy`: inject an already-loaded heavy (e.g. LLaDA2MoeModelLM.from_pretrained) to avoid a wasteful
        # random 16B init that would just be overwritten. Training (VeOmni meta path) leaves it None.
        self.heavy = _heavy if _heavy is not None else LLaDA2MoeModelLM(config)
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
        re-zero the Δh head. Always does the copy when CALLED (the `__init__` auto-call is gated by
        config.warmstart_from_heavy_bottom; the offline init builder calls this explicitly). NOTE: only valid
        when a heavy bottom layer is DENSE and the draft widths match the heavy; MoE heavy layers copy only
        their attention/norms (dense MLP stays init) — open decision in the design doc."""
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
    def extract_heavy_signals(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                              inputs_embeds: Optional[torch.Tensor] = None,
                              past_key_values=None, use_cache: bool = False) -> dict:
        """Run the FROZEN heavy once; capture hidden at config.sel_layers_list (concat -> m*D), the last-layer
        hidden, and the logits; split prefix/canvas. Returns the dict consumed by `draft_forward`. The heavy
        is bidirectional within a block, so a single pass over [prefix ; canvas] gives all signals.
        If `inputs_embeds` is given (DMax soft-embedding decode), the heavy forwards on it while the prefix/canvas
        split still uses the HARD `input_ids` (mask_token_id marks the canvas).
        PREFIX-KV CACHE (decode only): pass `past_key_values` (a DynamicCache holding the settled prefix [0,bs))
        + `use_cache=True` and feed ONLY the current block as `inputs_embeds` [1,blk,D] with an all-attend 4D
        `attention_mask` [1,1,blk,be]; the heavy computes just the block (position_ids default to arange(bs,be)
        from the cache length) and APPENDS its KV to the returned cache — the caller crops back to bs each iter,
        or keeps it on block completion. Then `logits/h_sel/h_last` are [1,blk,*] (block only)."""
        out = self.heavy(
            input_ids=None if inputs_embeds is not None else input_ids,
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            past_key_values=past_key_values, use_cache=use_cache,
            output_hidden_states=True, return_dict=True,
        )
        hs = out.hidden_states                                  # tuple of [B,N,D], len = num_layers+1
        sel = torch.cat([hs[i] for i in self.config.sel_layers_list], dim=-1)  # [B,N,m*D]
        h_last = hs[-1]                                         # [B,N,D]
        logits = out.logits                                    # [B,N,V]
        prefix_idx, canvas_idx = self._split_prefix_denoise(input_ids)
        return {
            "input_ids": input_ids, "logits": logits, "h_sel": sel, "h_last": h_last,
            "prefix_idx": prefix_idx, "canvas_idx": canvas_idx,
            "past_key_values": out.past_key_values if use_cache else None,
        }

    # ---- cross-block prefix-KV cache growth (decode) ----
    @torch.no_grad()
    def extend_draft_prefix_cache(self, h_sel_delta, position_ids, past_key_values):
        """Grow the drafter's cross-block prefix-KV cache by one just-settled block (see
        `DbetDraftStack.extend_prefix_cache`). Thin wrapper so the decode loop stays out of the model internals."""
        self.draft.extend_prefix_cache(h_sel_delta, position_ids, past_key_values)

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
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        denoise_mask: Optional[torch.Tensor] = None,
        tau: Optional[float] = None,
        dmax_ft_kwargs: Optional[dict] = None,
        dmax_ft_eval_kwargs: Optional[dict] = None,
    ) -> dict:
        """extract_heavy_signals -> draft_forward; returns the raw drafter outputs only. The model does NOT
        compute loss — the training script owns that (it has the labels, the loss/accept masks and the loss
        weighting). `attention_mask` (#1[+#2]) and `denoise_mask` (#4) are passed through.
        Returns dict: logits [B,C,V], conf [B,C] (or None), h_draft [B,C,D], delta [B,C,D].
        `dmax_ft_kwargs`: route to the HEAVY-fine-tune step (must go through forward so the FSDP root hook gathers
        the sharded embed/lm_head; direct submodule calls fail under FSDP2)."""
        if dmax_ft_kwargs is not None:
            return self.dmax_ft_forward(**dmax_ft_kwargs)
        if dmax_ft_eval_kwargs is not None:
            return self.dmax_ft_eval_forward(**dmax_ft_eval_kwargs)
        signals = self.extract_heavy_signals(input_ids, attention_mask)
        return self.draft_forward(
            signals, attention_mask=attention_mask, position_ids=position_ids,
            denoise_mask=denoise_mask, tau=tau,
        )

    def dmax_ft_forward(self, full, attention_mask, position_ids, noisy_len, mask_id,
                        heavy_thr, draft_k, heavy_top_k, heavy_tau, draft_tau, block_size):
        """HEAVY fine-tune merged 3-route step, run INSIDE forward so FSDP gathers the root params. Shares heavy
        fwd#1 (loss_B mask-denoise + commit + draft signals), then ONE batched B=2 second heavy forward:
        row0 = Route A (correct the draft rollout -> loss_A), row1 = Route C (all-heavy-commit self-refine ->
        loss_C, anti-degradation). Returns (loss_B, loss_A, loss_C, metrics). B(data)=1 assumed."""
        import torch.nn.functional as _F
        from dbet_train_core import heavy_commit as _heavy_commit, derive_drafter_mask as _derive_mask
        from dmax_dbet_train_core import soft_embed as _soft_embed

        cfg = self.config
        bs = block_size
        L = noisy_len
        noisy_ids, clean_ids = full[:, :L], full[:, L:]
        masked = (noisy_ids == mask_id)
        w = masked.float()
        denom = w.sum().clamp_min(1.0)
        embed = self.draft.frozen_embed

        def _ce(noisy_logits):
            ce = _F.cross_entropy(noisy_logits.reshape(-1, noisy_logits.shape[-1]).float(),
                                  clean_ids.reshape(-1), reduction="none").view_as(clean_ids)
            return (ce * w).sum() / denom

        # heavy fwd #1 (grad): mask-denoise loss + draft ingredients
        hout1 = self.heavy(input_ids=full, attention_mask=attention_mask, position_ids=position_ids,
                           use_cache=False, output_hidden_states=True, output_router_logits=False, return_dict=True)
        noisy_logits1 = hout1.logits[:, :L]
        loss_B = _ce(noisy_logits1)
        sel = torch.cat([hout1.hidden_states[i] for i in cfg.sel_layers_list], dim=-1)
        noisy_h_sel, clean_h_sel = sel[:, :L].detach(), sel[:, L:].detach()
        noisy_h_last = hout1.hidden_states[-1][:, :L].detach()
        noisy_logits1_d = noisy_logits1.detach()

        post_commit, remaining = _heavy_commit(noisy_logits1_d, noisy_ids, mask_id, bs, heavy_thr)
        heavy_committed = masked & (~remaining)

        with torch.no_grad():
            dout = self.draft(input_ids=post_commit, heavy_logits=noisy_logits1_d,
                              h_sel_denoise=noisy_h_sel, h_last_denoise=noisy_h_last, h_sel_prefix=clean_h_sel,
                              attention_mask=_derive_mask(attention_mask, L), position_ids=position_ids,
                              denoise_mask=None, tau=None)
        draft_logits = dout["logits"]

        # ---- build BOTH 2nd-pass noisy embeds (share revealed golden + heavy-committed soft-embed) ----
        base = embed(noisy_ids).detach()                        # golden@revealed, embed(MASK)@masked
        heavy_soft_committed = (_soft_embed(noisy_logits1_d[heavy_committed], embed, mask_id, heavy_tau, heavy_top_k)
                                if bool(heavy_committed.any()) else None)
        # Route A: golden ; heavy-soft(committed) ; DRAFT-soft(remaining)   (correct the draft rollout)
        embeds_A = base.clone()
        if heavy_soft_committed is not None:
            embeds_A[heavy_committed] = heavy_soft_committed
        if bool(remaining.any()):
            embeds_A[remaining] = _soft_embed(draft_logits[remaining], embed, mask_id, draft_tau, draft_k)
        # Route C: golden ; heavy-soft(ALL masked)  ("all commit" -> keep the heavy's own self-refine; anti-degrade)
        embeds_C = base.clone()
        if bool(masked.any()):
            embeds_C[masked] = _soft_embed(noisy_logits1_d[masked], embed, mask_id, heavy_tau, heavy_top_k)
        clean_embeds = embed(clean_ids).detach()

        # ---- ONE batched 2nd heavy forward (B=2): row0 = Route A, row1 = Route C (shares fwd#1; +1 fwd call) ----
        full_2 = torch.cat([torch.cat([embeds_A, clean_embeds], dim=1),
                            torch.cat([embeds_C, clean_embeds], dim=1)], dim=0)      # [2,2L,D]
        attn_2 = attention_mask.expand(2, *attention_mask.shape[1:]) if attention_mask is not None else None
        pos_2 = position_ids.expand(2, -1).contiguous() if position_ids is not None else None
        hout2 = self.heavy(inputs_embeds=full_2, attention_mask=attn_2, position_ids=pos_2,
                           use_cache=False, output_router_logits=False, return_dict=True)
        logits_A = hout2.logits[0:1, :L]                        # keep the [1,L,V] batch dim for _ce (vs clean_ids [1,L])
        logits_C = hout2.logits[1:2, :L]
        loss_A = _ce(logits_A)                                  # correct-draft route
        loss_C = _ce(logits_C)                                  # pure-heavy self-refine route

        metrics = {
            "loss_B": float(loss_B.detach()), "loss_A": float(loss_A.detach()), "loss_C": float(loss_C.detach()),
            "heavy_thr": heavy_thr, "draft_k": draft_k,
            "n_masked": int(masked.sum()), "n_heavy_commit": int(heavy_committed.sum()), "n_draft": int(remaining.sum()),
            "acc_heavy1": float(((noisy_logits1_d.argmax(-1) == clean_ids) & masked).float().sum() / denom),
            "acc_corr2": float(((logits_A.argmax(-1) == clean_ids) & masked).float().sum() / denom),
            "acc_corrC": float(((logits_C.argmax(-1) == clean_ids) & masked).float().sum() / denom),
        }
        return loss_B, loss_A, loss_C, metrics

    @torch.no_grad()
    def _heavy_multipass_acc(self, full, attention_mask, position_ids, noisy_len, mask_id,
                             heavy_thr, block_size, heavy_top_k, heavy_tau, n_passes):
        """Matched-budget PURE-HEAVY baseline: run the heavy `n_passes` times, decode_uniform-committing at
        `heavy_thr` (soft-embed re-feed) each pass; return accuracy on the originally-masked positions vs gold.
        n_passes=3 aligns with acc_corr2's 2-heavy+1-draft (3 model forwards). All no_grad."""
        from dbet_train_core import heavy_commit as _heavy_commit
        from dmax_dbet_train_core import soft_embed as _soft_embed
        L = noisy_len
        noisy_ids, clean_ids = full[:, :L], full[:, L:]
        active = (noisy_ids == mask_id)
        denom = active.float().sum().clamp_min(1.0)
        embed = self.draft.frozen_embed
        cur = noisy_ids.clone()                                  # committed tokens accumulate here (mask = uncommitted)
        noisy_embeds = embed(noisy_ids).clone()
        clean_embeds = embed(clean_ids)
        logits = None
        for p in range(n_passes):
            full_embeds = torch.cat([noisy_embeds, clean_embeds], dim=1)
            logits = self.heavy(inputs_embeds=full_embeds, attention_mask=attention_mask, position_ids=position_ids,
                                use_cache=False, output_router_logits=False, return_dict=True).logits[:, :L]
            if p < n_passes - 1:                                 # commit + soft-embed on all but the last pass
                post, _ = _heavy_commit(logits, cur, mask_id, block_size, heavy_thr)
                newly = (cur == mask_id) & (post != mask_id)
                cur = post
                if bool(newly.any()):
                    noisy_embeds[newly] = _soft_embed(logits[newly], embed, mask_id, heavy_tau, heavy_top_k)
        final = torch.where(cur == mask_id, logits.argmax(-1), cur)   # committed token, else last-pass argmax
        return float(((final == clean_ids) & active).float().sum() / denom)

    @torch.no_grad()
    def dmax_ft_eval_forward(self, full, attention_mask, position_ids, noisy_len, mask_id,
                             heavy_thr, draft_k, heavy_top_k, heavy_tau, draft_tau, block_size, n_heavy_passes):
        """Held-out val (run through forward for FSDP). Reuses dmax_ft_forward for loss_B/loss_A/acc_heavy1/
        acc_corr2 (2 heavy + draft), then adds acc_heavy3 (matched-budget pure-heavy). Returns the metrics dict."""
        _, _, _, m = self.dmax_ft_forward(full, attention_mask, position_ids, noisy_len, mask_id,
                                          heavy_thr, draft_k, heavy_top_k, heavy_tau, draft_tau, block_size)
        m["acc_heavy3"] = self._heavy_multipass_acc(full, attention_mask, position_ids, noisy_len, mask_id,
                                                    heavy_thr, block_size, heavy_top_k, heavy_tau, n_heavy_passes)
        return m


# VeOmni registry hook: `ModelRegistry.register_modeling_path("models.dbet")` walks SUBMODULES (pkgutil) and
# registers each module's `ModelClass` -> the package __init__ is never visited, so the export must live HERE
# (same convention as models/llada2_moe/modeling_llada2_moe.py). Without this, get_loader() falls back to the
# Huggingface AutoModel path, which fails on DbetForDraftDecoding (arch name lacks "ForCausalLM").
ModelClass = DbetForDraftDecoding
