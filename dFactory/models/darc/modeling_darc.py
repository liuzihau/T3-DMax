# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# DARC head: AR refinement chain grafted after layer `tap_layer` of a frozen DMax (LLaDA2-MoE) model.
# Mirrors dFactory/models/dbet/ (configuration_*.py + modeling_*.py). See ./README.md for the design.
#
# This file is SELF-CONTAINED (torch/nn only): the frozen backbone pieces (input embedding, lm_head,
# final_norm) and the RoPE cos/sin are PASSED IN, so the head's logic is unit-testable without the 16B model
# (run `python modeling_darc.py`). Base-model integration (import LLaDA2Moe pieces, build cos/sin from
# LLaDA2MoeRotaryEmbedding, tap via heavy_forward(output_hidden_states=True), and the Loss-2 top-layer
# replay) is the NEXT step -- marked [INTEGRATION] below.
#
# Loss-1 (this file): self-conditioned STOP-GRADIENT AR training. Per block, left-to-right:
#   * block-relative pos 0 = frozen SEED (no trainable params, NO loss); revealed positions = HARD embed.
#   * masked non-seed pos i: query h_i attends the DETACHED in-block prefix soft-embeds s_{bs..i-1}
#     -> MLP -> frozen readout -> CE(gold_i); then s_i = softembed_topk(readout).detach().
# Detach only changes the backward pass; forward values equal inference, so no exposure bias. The reference
# below is a single sequential pass (O(n) python steps). TODO(perf): block-parallel form -- blocks are
# independent, so loop only over the within-block index t in [1, block_size) vectorized over (batch·#blocks).

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from configuration_darc import DarcConfig


# ---- primitives (copies of the LLaDA-2.0 / DBet helpers; swap for the base-model imports at [INTEGRATION]) ----
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_single(x, cos, sin, unsqueeze_dim=1):
    """x [B,h,T,hd]; cos/sin [B,T,rot] -> [B,h,T,hd], partial rotary (rot may be < hd)."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rot = cos.shape[-1]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    return torch.cat([(x_rot * cos) + (rotate_half(x_rot) * sin), x_pass], dim=-1)


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


class DarcGatedMLP(nn.Module):
    """SwiGLU (gate/up/down named like LLaDA2MoeMLP). Optional pre-norm, zero-init out (Δ==0 at init),
    and an additive residual -- the DbetGatedMLP pattern."""

    def __init__(self, d_in, d_inter, d_out, *, act="silu", pre_norm=False, zero_init_out=False, eps=1e-6):
        super().__init__()
        self.pre_norm = RMSNorm(d_in, eps) if pre_norm else None
        self.gate_proj = nn.Linear(d_in, d_inter, bias=False)
        self.up_proj = nn.Linear(d_in, d_inter, bias=False)
        self.down_proj = nn.Linear(d_inter, d_out, bias=False)
        self.act_fn = F.silu if act == "silu" else getattr(F, act)
        if zero_init_out:
            nn.init.zeros_(self.down_proj.weight)

    def forward(self, x, residual_to=None):
        h = self.pre_norm(x) if self.pre_norm is not None else x
        y = self.down_proj(self.act_fn(self.gate_proj(h)) * self.up_proj(h))
        return y if residual_to is None else y + residual_to


class DarcARAttention(nn.Module):
    """Causal cross-attention: query from q_input (the tapped hidden h_i), keys/values from kv_input (the
    committed in-block soft-embeds). GQA + partial rotary + query/key RMSNorm, sdpa backend. The caller slices
    kv_input to exactly the allowed in-block prefix, so no attention mask is needed for the reference loop."""

    def __init__(self, config: DarcConfig):
        super().__init__()
        self.nH = config.num_attention_heads
        self.nKV = config.num_key_value_heads
        self.groups = self.nH // self.nKV
        self.hd = config.head_dim
        self.scale = self.hd ** -0.5
        D = config.hidden_size
        self.q_proj = nn.Linear(D, self.nH * self.hd, bias=False)
        self.kv_proj = nn.Linear(D, 2 * self.nKV * self.hd, bias=False)
        self.query_layernorm = RMSNorm(self.hd, config.rms_norm_eps)
        self.key_layernorm = RMSNorm(self.hd, config.rms_norm_eps)
        self.dense = nn.Linear(self.nH * self.hd, D, bias=False)

    def forward(self, q_input, kv_input, cos_q, sin_q, cos_k, sin_k, attn_mask=None):
        b, tq, _ = q_input.shape
        tk = kv_input.shape[1]
        q = self.q_proj(q_input).view(b, tq, self.nH, self.hd).transpose(1, 2)              # [B,H,Tq,hd]
        kv = self.kv_proj(kv_input).view(b, tk, 2 * self.nKV, self.hd).transpose(1, 2)
        k, v = kv.split([self.nKV, self.nKV], dim=1)                                          # [B,Hkv,Tk,hd]
        q = self.query_layernorm(q)
        k = self.key_layernorm(k)
        q = apply_rope_single(q, cos_q, sin_q)
        k = apply_rope_single(k, cos_k, sin_k)
        k = repeat_kv(k, self.groups)
        v = repeat_kv(v, self.groups)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False, scale=self.scale)
        return self.dense(out.transpose(1, 2).reshape(b, tq, -1))


class DarcFuse(nn.Module):
    """[Loss-2, added later] Fuse the soft-embed sequence back into residual space, zero-init residual so the
    L19 input == the base hidden at init (== base model). Not used by the Loss-1 forward below."""

    def __init__(self, config: DarcConfig):
        super().__init__()
        D = config.hidden_size
        self.mlp = DarcGatedMLP(2 * D, config.intermediate_size, D, act=config.hidden_act,
                                pre_norm=True, zero_init_out=True, eps=config.rms_norm_eps)

    def forward(self, soft_seq, h):
        return self.mlp(torch.cat([soft_seq, h], dim=-1), residual_to=h)


class DarcHead(nn.Module):
    """The trainable DARC modules + the Loss-1 forward. Frozen backbone pieces are passed in (not owned)."""

    def __init__(self, config: DarcConfig):
        super().__init__()
        self.config = config
        D = config.hidden_size
        self.input_layernorm = RMSNorm(D, config.rms_norm_eps)
        self.attention = DarcARAttention(config)
        self.post_attention_layernorm = RMSNorm(D, config.rms_norm_eps)
        self.mlp = DarcGatedMLP(D, config.intermediate_size, D, act=config.hidden_act,
                                pre_norm=False, eps=config.rms_norm_eps)
        self.fuse = DarcFuse(config)                          # for Loss-2 (unused here)

    # ---- soft-embed: top-k softmax-weighted FROZEN base embedding ----
    def soft_embed_topk(self, logits, embed_weight):
        k, tau = self.config.top_k, self.config.soft_tau
        topv, topi = logits.topk(k, dim=-1)                  # [...,k]
        w = torch.softmax(topv.float() / tau, dim=-1).to(embed_weight.dtype)
        e = embed_weight[topi]                               # [...,k,D]
        return (w.unsqueeze(-1) * e).sum(dim=-2)             # [...,D]

    def _ar_block(self, q_h, kv_s, cos_q, sin_q, cos_k, sin_k):
        residual = q_h
        h = self.input_layernorm(q_h)
        h = residual + self.attention(h, kv_s, cos_q, sin_q, cos_k, sin_k)
        return self.mlp(self.post_attention_layernorm(h), residual_to=h)

    @staticmethod
    def readout(ar_out, final_norm, lm_head):
        return lm_head(final_norm(ar_out))                   # standard logit lens (frozen)

    def forward_train(self, h, noisy_input_ids, labels, cos, sin, frozen_embed, final_norm, lm_head,
                      return_soft_embeds=False):
        """h [B,n,D] tapped hidden; noisy_input_ids/labels [B,n]; cos/sin [B,n,rot]. Returns (loss, metrics)."""
        B = self.config.block_size
        Bsz, n, D = h.shape
        dev = h.device
        embed_weight = frozen_embed.weight
        MASK = int(getattr(self.config, "mask_token_id", 156895))  # revealed vs masked from the noisy stream
        revealed = noisy_input_ids != MASK                   # True = a real/committed token (hard embed)

        # committed soft-embeds as a LIST (index == position), stacked per step -> a FRESH kv tensor each time,
        # so kv_proj's saved-for-backward input is never invalidated by a later in-place write.
        s_list = []                                          # each [B,D], detached
        ar_list, pos_list = [], []
        for i in range(n):
            bs = (i // B) * B
            cos_i, sin_i = cos[:, i:i + 1], sin[:, i:i + 1]
            if bool(revealed[:, i].all()):                   # (B=1 training assumed; .all() is exact then)
                s_list.append(frozen_embed(noisy_input_ids[:, i]).detach())
                continue
            if i == bs:                                       # masked SEED: direct logit-lens on h_i, no loss
                with torch.no_grad():
                    logit = self.readout(h[:, i:i + 1], final_norm, lm_head)
                    s_list.append(self.soft_embed_topk(logit, embed_weight).squeeze(1).detach())
                continue
            # masked NON-SEED: grad through the head; the in-block context is a detached, freshly-stacked tensor
            kv = torch.stack(s_list[bs:i], dim=1)            # [B, i-bs, D]
            ar = self._ar_block(h[:, i:i + 1], kv, cos_i, sin_i, cos[:, bs:i], sin[:, bs:i])  # [B,1,D]
            ar_list.append(ar)
            pos_list.append(i)
            with torch.no_grad():
                logit = self.readout(ar, final_norm, lm_head)
                s_list.append(self.soft_embed_topk(logit, embed_weight).squeeze(1).detach())

        if not ar_list:
            loss = h.sum() * 0.0
            metrics = {"loss1": 0.0, "n_sup": 0, "acc1": 0.0}
        else:
            AR = torch.cat(ar_list, dim=1)                                   # [B,nq,D]
            logits = self.readout(AR, final_norm, lm_head)                   # [B,nq,V]
            gold = labels[:, pos_list]                                       # [B,nq]
            V = logits.shape[-1]
            loss = F.cross_entropy(logits.reshape(-1, V), gold.reshape(-1), ignore_index=-100)
            with torch.no_grad():
                valid = gold != -100
                pred = logits.argmax(-1)
                acc1 = float((pred[valid] == gold[valid]).float().mean()) if bool(valid.any()) else 0.0
            metrics = {"loss1": float(loss.detach()), "n_sup": int((gold != -100).sum()), "acc1": acc1}
        if return_soft_embeds:
            return loss, metrics, torch.stack(s_list, dim=1)     # [B,n,D], detached
        return loss, metrics


# ============================================================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = DarcConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                     intermediate_size=128, vocab_size=200, rotary_dim=8, block_size=8, top_k=5)
    setattr(cfg, "mask_token_id", 199)
    D, V, n = cfg.hidden_size, cfg.vocab_size, 24          # 3 blocks of 8
    head = DarcHead(cfg).eval()
    frozen_embed = nn.Embedding(V, D)
    lm_head = nn.Linear(D, V, bias=False)
    final_norm = RMSNorm(D)
    for p in list(frozen_embed.parameters()) + list(lm_head.parameters()) + list(final_norm.parameters()):
        p.requires_grad_(False)

    # rope cos/sin for positions 0..n-1
    pos = torch.arange(n).float()
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.rotary_dim, 2).float() / cfg.rotary_dim))
    fr = torch.outer(pos, inv)
    emb = torch.cat([fr, fr], dim=-1)
    cos, sin = emb.cos()[None], emb.sin()[None]           # [1,n,rot]

    h = torch.randn(1, n, D)
    # first block: 4 revealed (prompt) + rest masked; blocks 2,3 fully masked
    noisy = torch.full((1, n), 199)
    noisy[0, :4] = torch.randint(0, 190, (4,))            # revealed prompt tokens
    labels = torch.randint(0, 190, (1, n))
    labels[0, :4] = -100                                  # prompt: no loss

    loss, m, s = head.forward_train(h, noisy, labels, cos, sin, frozen_embed, final_norm, lm_head,
                                    return_soft_embeds=True)
    print(f"loss={float(loss.detach()):.4f}  metrics={m}")
    assert torch.isfinite(loss), "loss must be finite (no NaN from empty attention rows)"

    # supervised count == non-seed masked positions (exclude seeds at 0,8,16 and the 4 revealed)
    seeds = {0, 8, 16}
    exp_sup = sum(1 for i in range(n) if i not in seeds and int(noisy[0, i]) == 199)
    assert m["n_sup"] == exp_sup, (m["n_sup"], exp_sup)
    print(f"[seed/mask] OK  supervised={m['n_sup']} (seeds & revealed excluded)")

    # causality: perturbing h in block 3 must NOT change soft-embeds of block 1
    h2 = h.clone(); h2[0, 20] += 5.0
    with torch.no_grad():
        _, _, s2 = head.forward_train(h2, noisy, labels, cos, sin, frozen_embed, final_norm, lm_head,
                                      return_soft_embeds=True)
    assert torch.allclose(s[0, :8], s2[0, :8], atol=1e-5), "cross-block leakage: block1 changed by block3"
    # within block: perturbing a LATER position must not change an earlier soft-embed
    h3 = h.clone(); h3[0, 14] += 5.0
    with torch.no_grad():
        _, _, s3 = head.forward_train(h3, noisy, labels, cos, sin, frozen_embed, final_norm, lm_head,
                                      return_soft_embeds=True)
    assert torch.allclose(s[0, :10], s3[0, :10], atol=1e-5), "acausal: earlier pos changed by a later pos"
    print("[causality] OK  no cross-block / no future leakage")

    # grad flows to head params, not to frozen backbone
    loss.backward()
    assert head.attention.q_proj.weight.grad is not None and head.mlp.down_proj.weight.grad is not None
    assert lm_head.weight.grad is None and frozen_embed.weight.grad is None
    print("[grad] OK  head trains, backbone frozen")
    print("modeling_darc self-test PASSED")
