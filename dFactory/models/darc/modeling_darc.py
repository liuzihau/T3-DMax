# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# DARC head: AR refinement chain grafted at hidden_states[tap_hidden_index] of a frozen DMax (LLaDA2-MoE)
# model (tap_hidden_index == probe-plot 'L{i}'; hs[i] = output of decoder layer i-1).
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
    """Loss-2 'soft' injection: fuse the (non-detached) top-k soft-embed + h back into residual space, zero-init
    residual so at init the injection == the base hidden (== base model). Bigger MLP (2D -> mult*D -> D) so it
    can map the embedding-space top-k prune into the residual manifold."""

    def __init__(self, config: DarcConfig):
        super().__init__()
        D = config.hidden_size
        inter = getattr(config, "fuse_hidden_mult", 6) * D
        self.mlp = DarcGatedMLP(2 * D, inter, D, act=config.hidden_act,
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
                                pre_norm=False, zero_init_out=True, eps=config.rms_norm_eps)
        # zero-init both block outputs -> at init ar_out == h, so readout == the base tap logit-lens and the
        # head starts AT the base model's recall and learns only the residual (DBet delta-head philosophy).
        nn.init.zeros_(self.attention.dense.weight)
        self.fuse = DarcFuse(config) if getattr(config, "loss2_inject", "ar_out") == "soft" else None

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
                      return_gen=False):
        """h [B,n,D] tapped hidden for ONE block; noisy_input_ids/labels [B,n]; cos/sin [B,n,rot].
        Returns (loss, metrics); if return_gen: also (gen_tap_logits [B,n_gen,V], gen_pos [local idxs],
        ar_out_seq [B,n_gen,D]) -- gen ordered by generated-index g_0,g_1,... . ar_out_seq is the AR block's
        refined hidden per generated position (g_0=h), injected as the new hs[tap] for Loss-2 (grad -> head)."""
        Bsz, n, D = h.shape
        embed_weight = frozen_embed.weight
        MASK = int(getattr(self.config, "mask_token_id", 156895))  # revealed vs masked from the noisy stream
        revealed = noisy_input_ids != MASK                   # True = a real/committed token (hard embed)

        # Per-position embeds for the fuse (index == position): hard for revealed, soft for generated (detached).
        # SEED = the FIRST GENERATED token g_0 (h->LM->soft, no attn+mlp, no trainable loss). g_1+ = AR attending
        # ONLY the generated soft-embeds so far (block-local; revealed prompt tail reaches the head via h, and
        # is fed to the fuse as a hard embed but NOT attended). g_0 is contiguous with g_1.. since the generated
        # region [r, be) is contiguous after the (revealed) prompt tail.
        s_list = []                                          # each [B,D], detached  (for the fuse)
        g0 = None                                            # local index of the first generated token (seed)
        ar_list, ar_pos = [], []                             # AR outputs (grad) + local positions, for g_1+
        seed_logit = None
        for i in range(n):
            if bool(revealed[:, i].all()):                   # (B=1 training assumed; .all() is exact then)
                s_list.append(frozen_embed(noisy_input_ids[:, i]).detach())        # hard embed (fuse; not attended)
                continue
            if g0 is None:                                    # g_0 = SEED: direct logit-lens on h_i, NO loss
                g0 = i
                with torch.no_grad():
                    seed_logit = self.readout(h[:, i:i + 1], final_norm, lm_head)   # [B,1,V]
                    s_list.append(self.soft_embed_topk(seed_logit, embed_weight).squeeze(1).detach())
                continue
            kv = torch.stack(s_list[g0:i], dim=1)             # generated soft-embeds g_0..g_{j-1}  [B,j,D]
            ar = self._ar_block(h[:, i:i + 1], kv, cos[:, i:i + 1], sin[:, i:i + 1],
                                cos[:, g0:i], sin[:, g0:i])   # [B,1,D] grad
            ar_list.append(ar); ar_pos.append(i)
            with torch.no_grad():
                s_list.append(self.soft_embed_topk(self.readout(ar, final_norm, lm_head),
                                                   embed_weight).squeeze(1).detach())

        if ar_list:                                          # trainable loss = CE over g_1+ (AR positions)
            AR = torch.cat(ar_list, dim=1)                   # [B,n_ar,D]
            logits_ar = self.readout(AR, final_norm, lm_head)   # [B,n_ar,V] grad
            gold_ar = labels[:, ar_pos]                      # [B,n_ar]
            V = logits_ar.shape[-1]
            loss = F.cross_entropy(logits_ar.reshape(-1, V), gold_ar.reshape(-1), ignore_index=-100)
            with torch.no_grad():
                v = gold_ar != -100
                acc1 = float((logits_ar.argmax(-1)[v] == gold_ar[v]).float().mean()) if bool(v.any()) else 0.0
                nsup = int(v.sum())
        else:
            loss = h.sum() * 0.0                             # block with only a seed (<=1 generated token)
            logits_ar, acc1, nsup = None, 0.0, 0
        metrics = {"loss1": float(loss.detach()), "acc1": acc1, "n_sup": nsup}

        if not return_gen:
            return loss, metrics
        # per-GENERATED-position tap logits (g_0 seed + g_1+ AR), ordered by generated-index -> position-wise metrics
        gen_pos = ([g0] if g0 is not None else []) + ar_pos                    # local positions g_0..g_{k-1}
        parts = ([seed_logit] if seed_logit is not None else []) + \
                ([logits_ar.detach()] if logits_ar is not None else [])
        gen_tap_logits = torch.cat(parts, dim=1) if parts else None           # [B,n_gen,V] detached
        # ar_out sequence for the Loss-2 injection: g_0 -> h (base, seed unrefined); g_1+ -> AR block output
        # (grad, NOT detached, so Loss-2 trains attn+mlp). At init (zero-init AR) ar_out == h -> identity.
        ar_parts = ([h[:, g0:g0 + 1]] if g0 is not None else []) + ar_list
        ar_out_seq = torch.cat(ar_parts, dim=1) if ar_parts else None         # [B,n_gen,D]
        return loss, metrics, gen_tap_logits, gen_pos, ar_out_seq


# ============================================================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = DarcConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                     intermediate_size=128, vocab_size=200, rotary_dim=8, block_size=8, top_k=5)
    setattr(cfg, "mask_token_id", 199)
    D, V = cfg.hidden_size, cfg.vocab_size
    head = DarcHead(cfg).eval()                            # note: head processes ONE block per call
    frozen_embed = nn.Embedding(V, D)
    lm_head = nn.Linear(D, V, bias=False)
    final_norm = RMSNorm(D)
    for p in list(frozen_embed.parameters()) + list(lm_head.parameters()) + list(final_norm.parameters()):
        p.requires_grad_(False)

    def rope(positions):
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.rotary_dim, 2).float() / cfg.rotary_dim))
        emb = torch.cat([torch.outer(positions.float(), inv)] * 2, dim=-1)
        return emb.cos()[None], emb.sin()[None]

    def run(h, noisy, labels, base_pos):
        cos, sin = rope(base_pos)
        return head.forward_train(h, noisy, labels, cos, sin, frozen_embed, final_norm, lm_head, return_gen=True)

    # --- FULL block (8 masked): g0=0 seed, g1..g7 AR ---
    B = 8
    h = torch.randn(1, B, D)
    noisy = torch.full((1, B), 199)
    labels = torch.randint(0, 190, (1, B))
    loss, m, gl, gp, s = run(h, noisy, labels, torch.arange(B))
    print(f"[full] loss={float(loss.detach()):.4f} metrics={m}")
    assert m["n_sup"] == B - 1 and gp == list(range(B)) and gl.shape == (1, B, V)   # seed g0 + 7 AR
    assert torch.isfinite(loss)

    # --- PARTIAL block (3 revealed prompt + 5 masked): g0=3 (first mask) seed, g4..g7 AR ---
    hp = torch.randn(1, B, D)
    noisyp = torch.full((1, B), 199); noisyp[0, :3] = torch.randint(0, 190, (3,))
    labelsp = torch.randint(0, 190, (1, B)); labelsp[0, :3] = -100
    lp, mp, glp, gpp, sp = run(hp, noisyp, labelsp, torch.arange(B))
    assert gpp == [3, 4, 5, 6, 7], gpp                     # g0 = first MASK token (pos 3), not rel-0
    assert mp["n_sup"] == 4 and glp.shape == (1, 5, V)     # seed g0(pos3) + 4 AR(pos4..7); loss over the 4
    print(f"[partial] OK  seed g0=pos{gpp[0]}  gen_pos={gpp}  n_sup={mp['n_sup']}")

    # causality: perturbing h at a LATER generated pos must not change an earlier ar_out
    h2 = h.clone(); h2[0, 6] += 5.0
    with torch.no_grad():
        _, _, _, _, s2 = run(h2, noisy, labels, torch.arange(B))
    assert torch.allclose(s[0, :5], s2[0, :5], atol=1e-5), "acausal: earlier ar_out changed by a later pos"
    print("[causality] OK  no future leakage")

    # grad -> head params, not the frozen backbone
    loss.backward()
    assert head.attention.q_proj.weight.grad is not None and head.mlp.down_proj.weight.grad is not None
    assert lm_head.weight.grad is None and frozen_embed.weight.grad is None
    print("[grad] OK  head trains, backbone frozen")
    print("modeling_darc self-test PASSED")
