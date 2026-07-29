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


class DarcARLayer(nn.Module):
    """One AR refinement layer: pre-norm cross-attention (Q from the pass's query hidden, K/V from the committed
    soft-embeds) + SwiGLU MLP, both zero-init at the output -> ar_out == query at init (identity)."""

    def __init__(self, config: DarcConfig):
        super().__init__()
        D = config.hidden_size
        self.input_layernorm = RMSNorm(D, config.rms_norm_eps)
        self.attention = DarcARAttention(config)
        self.post_attention_layernorm = RMSNorm(D, config.rms_norm_eps)
        self.mlp = DarcGatedMLP(D, config.intermediate_size, D, act=config.hidden_act,
                                pre_norm=False, zero_init_out=True, eps=config.rms_norm_eps)
        nn.init.zeros_(self.attention.dense.weight)

    def forward(self, q_h, kv_s, cos_q, sin_q, cos_k, sin_k):
        residual = q_h
        h = self.input_layernorm(q_h)
        h = residual + self.attention(h, kv_s, cos_q, sin_q, cos_k, sin_k)
        return self.mlp(self.post_attention_layernorm(h), residual_to=h)


class DarcHead(nn.Module):
    """Trainable DARC modules + Loss-1 forward. n_ar_passes stacked AR layers (pass p's queries = pass p-1's
    outputs; pass-1 dynamic-k, later passes fixed top_k). Frozen backbone pieces are passed in (not owned)."""

    def __init__(self, config: DarcConfig):
        super().__init__()
        self.config = config
        n = max(1, int(getattr(config, "n_ar_passes", 1)))
        self.ar_layers = nn.ModuleList([DarcARLayer(config) for _ in range(n)])   # stacked AR passes
        self.fuse = DarcFuse(config) if getattr(config, "loss2_inject", "ar_out") == "soft" else None

    def _k_at(self, pos, pass_idx):                          # PASS-1 dynamic-k by position; else fixed top_k
        if pass_idx == 0 and getattr(self.config, "dynamic_k", False):
            for upper, k in self.config.k_schedule:
                if pos < upper:
                    return int(k)
            return int(self.config.k_schedule[-1][1])
        return int(self.config.top_k)

    # ---- soft-embed: top-k softmax-weighted FROZEN base embedding ----
    def soft_embed_topk(self, logits, embed_weight, k=None):
        k = int(k if k is not None else self.config.top_k)
        topv, topi = logits.topk(k, dim=-1)                  # [...,k]
        w = torch.softmax(topv.float() / self.config.soft_tau, dim=-1).to(embed_weight.dtype)
        return (w.unsqueeze(-1) * embed_weight[topi]).sum(dim=-2)   # [...,D]

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
        ew = frozen_embed.weight
        MASK = int(getattr(self.config, "mask_token_id", 156895))
        revealed = noisy_input_ids != MASK                   # True = a real/committed token (hard embed)

        # N stacked AR passes over the generated positions. Per pass: SEED (first generated g_0) output = the
        # pass's query (no AR, no loss); g_1+ = AR attending ONLY the generated soft-embeds so far (detached ->
        # per-pass stop-gradient). Pass p's query = pass p-1's refined output (grad flows across passes). PASS-1
        # uses dynamic-k soft-embeds; later passes fixed top_k.
        g0 = None                                            # first generated local index (same across passes)
        q_seq = h                                            # pass-1 queries = tap hidden; refined each pass
        for pi, layer in enumerate(self.ar_layers):
            s_list = []                                      # this pass's soft-embeds (index==pos), detached
            ar_list = []
            g0 = None
            for i in range(n):
                if bool(revealed[:, i].all()):               # (B=1 training; .all() exact)
                    s_list.append(frozen_embed(noisy_input_ids[:, i]).detach())    # hard embed (context source)
                    continue
                if g0 is None:                               # g_0 = SEED: output = query (no AR), no loss
                    g0 = i
                    with torch.no_grad():
                        lg = self.readout(q_seq[:, i:i + 1], final_norm, lm_head)
                        s_list.append(self.soft_embed_topk(lg, ew, self._k_at(i, pi)).squeeze(1).detach())
                    continue
                kv = torch.stack(s_list[g0:i], dim=1)        # generated soft-embeds so far
                ar = layer(q_seq[:, i:i + 1], kv, cos[:, i:i + 1], sin[:, i:i + 1], cos[:, g0:i], sin[:, g0:i])
                ar_list.append(ar)                           # [B,1,D] grad
                with torch.no_grad():
                    lg = self.readout(ar, final_norm, lm_head)
                    s_list.append(self.soft_embed_topk(lg, ew, self._k_at(i, pi)).squeeze(1).detach())
            if ar_list:                                      # next pass's queries: revealed+seed keep q_seq;
                q_seq = torch.cat([q_seq[:, :g0 + 1], torch.cat(ar_list, dim=1)], dim=1)  # g_1+ = this pass output

        if g0 is None or n - g0 <= 1:                        # no generated tokens, or only a seed -> no loss
            loss = h.sum() * 0.0
            metrics = {"loss1": 0.0, "acc1": 0.0, "n_sup": 0}
            return (loss, metrics, None, [], None) if return_gen else (loss, metrics)

        gen_out = q_seq[:, g0:]                               # [B,n_gen,D] final refined (g_0 seed == h[g0])
        gen_logits = self.readout(gen_out, final_norm, lm_head)   # [B,n_gen,V]  ([:,0]=seed no-grad; [:,1:] grad)
        gold_g = labels[:, g0:]                              # [B,n_gen]
        V = gen_logits.shape[-1]
        loss = F.cross_entropy(gen_logits[:, 1:].reshape(-1, V), gold_g[:, 1:].reshape(-1), ignore_index=-100)
        with torch.no_grad():
            gv = gold_g[:, 1:] != -100
            acc1 = float((gen_logits[:, 1:].argmax(-1)[gv] == gold_g[:, 1:][gv]).float().mean()) \
                if bool(gv.any()) else 0.0
            nsup = int(gv.sum())
        metrics = {"loss1": float(loss.detach()), "acc1": acc1, "n_sup": nsup}
        if not return_gen:
            return loss, metrics
        return loss, metrics, gen_logits.detach(), list(range(g0, n)), gen_out


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
    assert head.ar_layers[0].attention.q_proj.weight.grad is not None
    assert head.ar_layers[0].mlp.down_proj.weight.grad is not None
    assert lm_head.weight.grad is None and frozen_embed.weight.grad is None
    print("[grad] OK  head trains, backbone frozen")

    # 2-pass + dynamic-k: grad reaches BOTH passes (Loss trains pass-1 through the query chain)
    cfg2 = DarcConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                      intermediate_size=128, vocab_size=200, rotary_dim=8, block_size=32, top_k=5,
                      n_ar_passes=2, dynamic_k=True)
    setattr(cfg2, "mask_token_id", 199)
    head2 = DarcHead(cfg2)
    assert len(head2.ar_layers) == 2 and head2._k_at(2, 0) == 5 and head2._k_at(10, 0) == 25 \
        and head2._k_at(30, 0) == 100 and head2._k_at(10, 1) == 5   # pass-2 uses fixed top_k
    B2 = 32
    h4 = torch.randn(1, B2, D); noisy4 = torch.full((1, B2), 199); labels4 = torch.randint(0, 190, (1, B2))
    cos4, sin4 = rope(torch.arange(B2))
    l4, m4, gl4, gp4, ao4 = head2.forward_train(h4, noisy4, labels4, cos4, sin4, frozen_embed, final_norm,
                                                lm_head, return_gen=True)
    l4.backward()
    assert head2.ar_layers[0].attention.q_proj.weight.grad is not None, "pass-1 not trained (grad broke)"
    assert head2.ar_layers[1].attention.q_proj.weight.grad is not None, "pass-2 not trained"
    assert gp4 == list(range(B2)) and ao4.shape == (1, B2, D)
    print("[2-pass+dynk] OK  both passes get grad; dynamic-k schedule 5/10/25/50/100")
    print("modeling_darc self-test PASSED")
