"""Top-K soft-embedding for the T3-D top-K talk trial.

The new core object of the trial (see probe_runner/T3D_TOPK_TALK_PLAN.md). For each
position, blend the *input* embeddings of think's top-K candidate tokens, weighted by
their probabilities, and rescale to the embedding-norm manifold. This is the talk's
input for still-undecided positions (replacing the bare [MASK]) and the rollout
feedback during training.

Two variants, one function:
  * INFERENCE (keep_mask_residual=True): keeps (1 - sum top-K prob) mass on [MASK],
    matching dInfer ParallelStrategy.decode_uniform's soft-embed BYTE-FOR-BYTE so
    train/inference agree (uncertain positions hedge toward mask).
  * TRAINING  (keep_mask_residual=False): drops the mask residual, renormalizes the
    weights within the top-K (w_i = p_i / sum_topk p) -- the OPUT-aligned "the answer
    is one of these K" feedback the plan specifies.

IMPORTANT (the model is UNTIED: tie_word_embeddings=false): build the blend from the
INPUT embedding table (`get_input_embeddings()` / `word_embeddings`), and take the
top-K + probabilities from the lm_head logits. These are different matrices here.

Unit-tested via `python t3d_topk_soft_embed.py` (CPU, tiny synthetic).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _embed(embedding, ids):
    """Embed token ids with an nn.Embedding module or a [V, D] weight tensor."""
    if isinstance(embedding, torch.Tensor):
        return F.embedding(ids, embedding)
    return embedding(ids)


def mask_embed_and_norm(embedding, mask_token_id: int, device, dtype):
    """The [MASK] input embedding [1,1,D] and its L2 norm [1,1,1]."""
    mid = torch.tensor([[mask_token_id]], device=device, dtype=torch.long)
    e = _embed(embedding, mid).to(dtype)                    # [1,1,D]
    return e, e.norm(p=2, dim=-1, keepdim=True)


@torch.no_grad()
def build_topk_soft_embeds(
    logits: torch.Tensor,           # [B, L, V] think's logits
    embedding,                      # input-embedding nn.Module or [V, D] weight
    mask_token_id: int,
    *,
    top_k: int = 10,
    keep_mask_residual: bool = True,   # True = inference (mask hedge); False = training
    use_float64: bool = False,
) -> torch.Tensor:
    """Return soft input-embeddings [B, L, D].

    Matches decode_uniform's soft-embed exactly when keep_mask_residual=True.
    """
    pdtype = torch.float64 if use_float64 else torch.float32
    probs = F.softmax(logits.to(pdtype), dim=-1)                    # [B,L,V]
    topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)         # [B,L,K]

    if keep_mask_residual:
        weights = topk_probs                                       # leave mass for [MASK]
        residual = torch.clamp(1.0 - topk_probs.sum(-1, keepdim=True), min=0.0)  # [B,L,1]
    else:
        weights = topk_probs / topk_probs.sum(-1, keepdim=True).clamp_min(1e-12)  # renorm in top-K
        residual = torch.zeros_like(topk_probs[..., :1])

    out_dtype = (embedding.weight.dtype if hasattr(embedding, "weight")
                 else embedding.dtype if isinstance(embedding, torch.Tensor)
                 else logits.dtype)
    topk_embeds = _embed(embedding, topk_idx).to(pdtype)           # [B,L,K,D]
    mask_embed, mask_norm = mask_embed_and_norm(embedding, mask_token_id, logits.device, pdtype)

    topk_weighted = (topk_embeds * weights.unsqueeze(-1)).sum(dim=2)   # [B,L,D]
    soft = topk_weighted + mask_embed * residual                      # [B,L,D]

    # rescale to the embedding-norm manifold (same as decode_uniform)
    cur_norm = soft.norm(p=2, dim=-1, keepdim=True)                    # [B,L,1]
    topk_norms = topk_embeds.norm(p=2, dim=-1)                         # [B,L,K]
    target_norm = (topk_norms * weights).sum(-1, keepdim=True) + mask_norm * residual
    soft = soft * (target_norm / (cur_norm + 1e-6))
    return soft.to(out_dtype)


def inject_soft_embeds(inputs_embeds, soft_embeds, positions):
    """Overwrite `inputs_embeds` at `positions` (bool [B,L]) with `soft_embeds`.

    Use this to feed the top-K blend to the talk for still-undecided positions,
    instead of the [MASK] (mask path) or argmax-token (old rollout) embedding."""
    out = inputs_embeds.clone()
    out[positions] = soft_embeds[positions].to(out.dtype)
    return out


# --------------------------------------------------------------------------- test
def _selftest():
    torch.manual_seed(0)
    B, L, V, D, K = 2, 4, 50, 8, 10
    W = torch.randn(V, D)                       # untied "input embedding" table
    mask_id = V - 1
    logits = torch.randn(B, L, V)

    # inference variant: must match a direct reimpl of decode_uniform's math
    soft_inf = build_topk_soft_embeds(logits, W, mask_id, top_k=K, keep_mask_residual=True)
    assert soft_inf.shape == (B, L, D)
    # reference (decode_uniform algebra)
    probs = F.softmax(logits.float(), -1)
    tp, ti = torch.topk(probs, K, -1)
    res = (1 - tp.sum(-1, keepdim=True)).clamp_min(0)
    te = F.embedding(ti, W)
    me = W[mask_id].view(1, 1, D); mn = me.norm(dim=-1, keepdim=True)
    s = (te * tp.unsqueeze(-1)).sum(2) + me * res
    tn = (te.norm(dim=-1) * tp).sum(-1, keepdim=True) + mn * res
    s = s * (tn / (s.norm(dim=-1, keepdim=True) + 1e-6))
    assert torch.allclose(soft_inf, s, atol=1e-4), (soft_inf - s).abs().max()

    # training variant: no mask mass; weights renormalize within top-K
    soft_tr = build_topk_soft_embeds(logits, W, mask_id, top_k=K, keep_mask_residual=False)
    # with K==V the renormalized weights == full softmax; sanity: finite, right shape
    assert soft_tr.shape == (B, L, D) and torch.isfinite(soft_tr).all()
    # the two variants differ (mask residual present vs not) unless top-K mass == 1
    assert not torch.allclose(soft_inf, soft_tr, atol=1e-3)

    # injection helper
    ie = torch.zeros(B, L, D)
    pos = torch.zeros(B, L, dtype=torch.bool); pos[0, 1] = True
    out = inject_soft_embeds(ie, soft_tr, pos)
    assert torch.allclose(out[0, 1], soft_tr[0, 1].to(out.dtype)) and out[0, 0].abs().sum() == 0

    # works with an nn.Embedding too
    emb = torch.nn.Embedding(V, D); emb.weight.data = W.clone()
    soft_mod = build_topk_soft_embeds(logits, emb, mask_id, top_k=K, keep_mask_residual=True)
    assert torch.allclose(soft_mod, soft_inf, atol=1e-4)
    print("t3d_topk_soft_embed selftest OK")


if __name__ == "__main__":
    _selftest()
