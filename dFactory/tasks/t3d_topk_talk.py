"""Anchor-free top-K talk — core input construction (T3-D top-K talk trial).

The new idea (see probe_runner/T3D_TOPK_TALK_PLAN.md + T3D_TOPK_TALK_INTEGRATION.md):
drop the hidden-state anchor entirely; the talk's ONLY signal from think is the
**top-K candidate set**, injected as the input embedding at still-masked positions.

This module is the well-defined, testable piece:
  build_talk_inputs_embeds(noisy, think_logits, embedding, mask_id, mode=...) -> [B,L,D]
    * committed positions (noisy != mask) -> their token's input embedding (context)
    * still-masked positions (noisy == mask):
        mode='mask'      -> the [MASK] embedding            (Path A, base/regularizer)
        mode='topk_soft' -> think's top-K soft-embedding    (Path B, the new ingredient)

`think_logits = lm_head(think_last_hidden)`. Untied model: the soft-embed is built
from the INPUT embedding table (handled inside build_topk_soft_embeds).

The talk then runs ANCHOR-FREE on these embeds (talk_model(inputs_embeds=..., anchor=None)),
which requires a talk config with anchor conditioning + cross-attention DISABLED.

Self-test: `python t3d_topk_talk.py` (CPU, tiny synthetic).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    from .t3d_topk_soft_embed import build_topk_soft_embeds, inject_soft_embeds   # package import
except ImportError:                                                  # standalone `python t3d_topk_talk.py`
    from t3d_topk_soft_embed import build_topk_soft_embeds, inject_soft_embeds


def _embed(embedding, ids):
    if isinstance(embedding, torch.Tensor):
        return F.embedding(ids, embedding)
    return embedding(ids)


def load_causal_lm(path, device, dtype=torch.bfloat16, attn_implementation="sdpa"):
    """Load an LLaDA2-Moe causal LM. Mirrors probe_runner.load_llada2's recipe:
    dFactory-class fallback + `moe_implementation='fused'` + `_veomni` model_type —
    WITHOUT which HF re-inits all ~16B params per-expert (the slow `normal_` hang).
    Used for the FROZEN think model in the top-K talk training (the trainable talk is
    still built by VeOmni's build_foundation_model)."""
    from transformers import AutoModelForCausalLM, AutoConfig
    try:
        m = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=dtype, attn_implementation=attn_implementation)
    except Exception as exc:
        print(f"[t3d] AutoModelForCausalLM failed ({type(exc).__name__}); dFactory class fallback")
        from models.llada2_moe.modeling_llada2_moe import LLaDA2MoeModelLM  # type: ignore
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        if not str(getattr(cfg, "model_type", "")).endswith("_veomni"):
            cfg.model_type = str(cfg.model_type) + "_veomni"
        if getattr(cfg, "moe_implementation", None) != "fused":
            cfg.moe_implementation = "fused"
        m = LLaDA2MoeModelLM.from_pretrained(
            path, config=cfg, torch_dtype=dtype, attn_implementation=attn_implementation)
    if hasattr(getattr(m, "model", None), "gradient_checkpointing"):
        m.model.gradient_checkpointing = False
    return m.to(device).eval()


def set_talk_trainable(talk, train_layer_idx, *, freeze_embed_head=True):
    """Freeze the whole talk, then unfreeze only the decoder layers in `train_layer_idx`.
    For 'merged layers only' on a keep=0-5,12,19 / n_merged=1 talk, the two merged-
    representative layers are at stack positions **6 and 8** (0-5 = kept layers 0-5,
    6 = rep(6-11), 7 = layer 12, 8 = rep(13-18), 9 = layer 19). embed/head stay frozen.
    Returns the number of trainable params."""
    for p in talk.parameters():
        p.requires_grad_(False)
    base = getattr(talk, "model", talk)
    layers = base.layers
    for i in train_layer_idx:
        for p in layers[i].parameters():
            p.requires_grad_(True)
    if not freeze_embed_head:
        for p in talk.get_input_embeddings().parameters():
            p.requires_grad_(True)
        head = talk.get_output_embeddings() or getattr(talk, "lm_head", None)
        if head is not None:
            for p in head.parameters():
                p.requires_grad_(True)
    return sum(p.numel() for p in talk.parameters() if p.requires_grad)


def build_talk_inputs_embeds(
    noisy: torch.Tensor,            # [B, L] current token ids (mask_id where undecided)
    think_logits: torch.Tensor,     # [B, L, V] = lm_head(think last hidden), no grad
    embedding,                      # input-embedding nn.Module or [V, D] weight
    mask_id: int,
    *,
    mode: str = "topk_soft",        # 'mask' (Path A) | 'topk_soft' (Path B)
    top_k: int = 10,
    keep_mask_residual: bool = False,  # training: False (renorm in top-K); inference: True
) -> torch.Tensor:
    """The talk's inputs_embeds. Committed positions keep their hard token embedding;
    still-masked positions get either [MASK] (mode='mask') or think's top-K soft-embed
    (mode='topk_soft'). No anchor anywhere."""
    base = _embed(embedding, noisy)                       # committed->token, masked->[MASK]
    if mode == "mask":
        return base
    if mode != "topk_soft":
        raise ValueError(f"unknown mode {mode!r}")
    masked = (noisy == mask_id)                           # [B, L]
    if not bool(masked.any()):
        return base
    soft = build_topk_soft_embeds(
        think_logits, embedding, mask_id, top_k=top_k,
        keep_mask_residual=keep_mask_residual)            # [B, L, D]
    out = base.clone()
    out[masked] = soft[masked].to(out.dtype)
    return out


def think_distill_loss(talk_logits: torch.Tensor, think_logits: torch.Tensor,
                       labels: torch.Tensor, *, temperature: float = 1.0,
                       weight: "torch.Tensor | None" = None,
                       ignore_index: int = -100, eps: float = 1e-8,
                       reverse: bool = False, kl_top_k: int = -1) -> torch.Tensor:
    """think<->talk distillation at the predict positions (labels != ignore_index). Two directions:

    * `reverse=False` (default): FORWARD KL  D_KL(think || talk)  (== soft cross-entropy up to
      think's entropy). MASS-COVERING: pulls the talk to cover think's full top-K, not just its
      top-1. The right signal for the STAGE-1 cold start.
    * `reverse=True`: REVERSE KL  D_KL(talk || think). MODE-SEEKING: pulls the talk onto think's
      high-probability mode. This is the STAGE-2 on-policy OPD objective (THUNLP 2604) -- combined
      with gold CE the split is clean: reverse-KL -> parity (H1), gold CE -> beat-think (H2).

    `kl_top_k>0` restricts the divergence to think's TOP-K candidate support (gather think's top-k
    ids, renormalize BOTH sides over that set via log_softmax) -- 2604 shows overlap/top-k KL ~=
    full-vocab at far lower cost; `-1` = full vocab. `temperature` softens both sides (loss x T^2).
    `weight` (optional [B,L]) -> weighted mean over predict positions. Returns a scalar (0 at a
    perfect match). fp32; think_logits and weight must be aligned to talk_logits (no shift)."""
    predict = (labels != ignore_index)                        # [B, L] bool
    if not bool(predict.any()):
        return talk_logits.sum() * 0.0                        # keep graph, zero contribution
    T = float(temperature)
    th = (think_logits[predict].float() / T)                  # [N, V]
    tl = (talk_logits[predict].float() / T)                   # [N, V]
    if kl_top_k and kl_top_k > 0 and kl_top_k < th.shape[-1]:
        _idx = th.topk(kl_top_k, dim=-1).indices              # think's top-k candidate support
        th = th.gather(-1, _idx); tl = tl.gather(-1, _idx)    # restrict both to that set
    teacher_logp = F.log_softmax(th, dim=-1)
    student_logp = F.log_softmax(tl, dim=-1)
    if reverse:                                               # D(talk || think): mode-seeking
        student = student_logp.exp()
        kl = (student * (student_logp - teacher_logp)).sum(-1)
    else:                                                     # D(think || talk): mass-covering
        teacher = teacher_logp.exp()
        kl = (teacher * (teacher_logp - student_logp)).sum(-1)
    if weight is None:
        return kl.mean() * (T * T)
    w = weight[predict].float()                               # [N]
    return (kl * w).sum() / w.sum().clamp_min(eps) * (T * T)  # weighted mean


def think_entropy_norm(logits: torch.Tensor) -> torch.Tensor:
    """Per-position NORMALIZED entropy of softmax(logits), in [0,1] (H / log V). High = the
    think model is uncertain there = forking/hard. Used to up-weight hard positions (Path A)."""
    import math
    p = F.softmax(logits.float(), dim=-1)
    H = -(p * p.clamp_min(1e-12).log()).sum(-1)               # [B, L]
    return H / math.log(logits.shape[-1])


def recoverable_mask(think_logits: torch.Tensor, labels: torch.Tensor, top_k: int,
                     ignore_index: int = -100) -> torch.Tensor:
    """[B,L] bool: a 'recoverable hard' position for Path B = think's top-1 is WRONG but the
    gold token IS in think's top-K (so the talk, fed the top-K, *can* pick it). Excludes
    non-predict positions. Where gold is NOT in top-K, Path B can't recover -> left to Path A."""
    predict = labels != ignore_index
    argmax = think_logits.argmax(dim=-1)                       # think top-1
    topk_idx = think_logits.topk(top_k, dim=-1).indices        # [B, L, K]
    gold_in_topk = (topk_idx == labels.unsqueeze(-1)).any(dim=-1)
    return predict & (argmax != labels) & gold_in_topk


def confident_prefix_commit(logits, mask_index, block_size, threshold):
    """Per-block confident LEFT-TO-RIGHT prefix commit — the DMax decode_uniform rule,
    generalized from one block to the whole noisy half [B, L]. Commits the contiguous
    prefix of masked positions whose softmax peak > threshold within each block (non-mask
    positions never break the cutoff). Matches the inference commit pattern, so the talk
    trains on the same own-commit contexts it faces at decode time.

    Returns (argmax [B, L], commit_mask [B, L] bool)."""
    B, L, V = logits.shape
    argmax = logits.argmax(dim=-1)                                   # [B, L]
    probs = F.softmax(logits.float(), dim=-1)
    max_probs = probs.gather(-1, argmax.unsqueeze(-1)).squeeze(-1)   # [B, L]
    # masked positions gate the cutoff; non-masked positions (=1.0) never break the prefix
    conf = torch.where(mask_index, max_probs, torch.ones_like(max_probs))
    nb = L // block_size
    confident = (conf > threshold).view(B, nb, block_size).long()
    prefix = torch.cumprod(confident, dim=-1).bool().view(B, L)      # 1 until first low-conf in block
    return argmax, (mask_index & prefix)


def build_think_next_teacher(think, think_emb, full_input_ids, think_s1_logits, noisy_len,
                             mask_id, *, block_size, threshold,
                             attention_mask=None, position_ids=None):
    """The think_{s+2} teacher for the Path-B (stage-1) KL.

    Advances the FROZEN think ONE DMax decode step from its s+1 state and returns the
    resulting next-iteration logits (the distillation target). Steps, on the noisy half:
      1. DMax decode of think's s+1 logits -> `confident_prefix_commit` (left-to-right,
         threshold-gated) -> (argmax, commit_mask).
      2. Rebuild think's next input the `decode_uniform` way: newly-committed positions get
         the SOFT mix of the SINGLE committed token blended with [MASK], renormalized
         (top_k=1 -- think commits ONE token per position, so this is its native commit
         representation, NOT the talk's top-10 input); reveal/earlier-committed stay their
         token embedding; still-masked stay bare [MASK]. The clean half is untouched.
      3. Re-forward think -> s+2 logits.

    Returns (s2_logits[:, :noisy_len], argmax, commit_mask). The caller reuses the SAME
    (argmax, commit_mask) to hard-commit Portion 1 into the talk's input, so one decode
    decision drives both the teacher and the talk's committed context. All under no_grad
    (think frozen). Build the teacher BEFORE writing Portion 1 into `full_input_ids`, so the
    commit is computed from the original s+1 (masked) state."""
    L = noisy_len
    noisy_ids = full_input_ids[:, :L]
    s1_noisy = think_s1_logits[:, :L]
    mask_index = (noisy_ids == mask_id)
    argmax, commit_mask = confident_prefix_commit(s1_noisy, mask_index, block_size, threshold)

    committed_ids = torch.where(commit_mask, argmax, noisy_ids)          # commit -> token, else unchanged
    noisy_in = _embed(think_emb, committed_ids)                          # reveal/commit -> token, mask -> [MASK]
    if bool(commit_mask.any()):
        # top_k=1: think's native commit soft-mix (committed token + mask residual, renorm) --
        # NOT the talk's top-10 input. This makes the teacher think's genuine 2nd iteration.
        soft = build_topk_soft_embeds(s1_noisy, think_emb, mask_id, top_k=1,
                                      keep_mask_residual=True)           # decode_uniform soft mix
        noisy_in = inject_soft_embeds(noisy_in, soft, commit_mask)       # newly-committed -> soft mix
    clean_in = _embed(think_emb, full_input_ids[:, L:])
    think_in = torch.cat([noisy_in, clean_in], dim=1).to(noisy_in.dtype)

    logits = think(inputs_embeds=think_in, attention_mask=attention_mask,
                   position_ids=position_ids, use_cache=False).logits
    return logits[:, :L], argmax, commit_mask


# --------------------------------------------------------------------------- test
def _selftest():
    torch.manual_seed(0)
    B, L, V, D, K = 2, 6, 40, 8, 10
    W = torch.randn(V, D)
    mask_id = V - 1
    noisy = torch.randint(0, V - 1, (B, L))          # committed tokens
    masked = torch.zeros(B, L, dtype=torch.bool)
    masked[0, 2:5] = True; masked[1, 0:2] = True      # some still-masked
    noisy[masked] = mask_id
    think_logits = torch.randn(B, L, V)

    # mask path = plain embedding of noisy (mask at masked positions)
    base = build_talk_inputs_embeds(noisy, think_logits, W, mask_id, mode="mask")
    assert torch.allclose(base, F.embedding(noisy, W))
    assert torch.allclose(base[masked], W[mask_id].expand(masked.sum(), D))   # masked -> [MASK]

    # topk_soft path = think's top-K at masked, token embeds at committed
    ts = build_talk_inputs_embeds(noisy, think_logits, W, mask_id, mode="topk_soft",
                                  top_k=K, keep_mask_residual=False)
    assert torch.allclose(ts[~masked], base[~masked])                 # committed unchanged
    assert not torch.allclose(ts[masked], base[masked])               # masked changed (top-K)
    soft = build_topk_soft_embeds(think_logits, W, mask_id, top_k=K, keep_mask_residual=False)
    assert torch.allclose(ts[masked], soft[masked].to(ts.dtype))      # exactly the top-K blend

    # no masked positions -> returns base untouched
    full = torch.randint(0, V - 1, (B, L))
    assert torch.allclose(build_talk_inputs_embeds(full, think_logits, W, mask_id, mode="topk_soft"),
                          F.embedding(full, W))

    talk_logits = torch.randn(B, L, V)
    labels = torch.randint(0, V - 1, (B, L))

    # think_distill_loss: forward KL at predict positions (labels!=-100)
    dl_labels = labels.clone(); dl_labels[~masked] = -100
    think_l = torch.randn(B, L, V)
    kl = think_distill_loss(talk_logits, think_l, dl_labels)
    assert torch.isfinite(kl) and kl > 0
    # KL(think||think) == 0 (perfect match), and temperature scaling stays finite
    assert think_distill_loss(think_l, think_l, dl_labels).abs() < 1e-5
    assert torch.isfinite(think_distill_loss(talk_logits, think_l, dl_labels, temperature=2.0))
    # REVERSE KL D(talk||think): finite, >0, and 0 at a perfect match (both directions vanish)
    rk = think_distill_loss(talk_logits, think_l, dl_labels, reverse=True)
    assert torch.isfinite(rk) and rk > 0
    assert think_distill_loss(think_l, think_l, dl_labels, reverse=True).abs() < 1e-5
    # kl_top_k truncation: finite, >0, and == full when kl_top_k >= V (no truncation)
    assert torch.isfinite(think_distill_loss(talk_logits, think_l, dl_labels, reverse=True, kl_top_k=10))
    assert torch.allclose(think_distill_loss(talk_logits, think_l, dl_labels, kl_top_k=V + 5),
                          think_distill_loss(talk_logits, think_l, dl_labels), atol=1e-5)
    # no predict positions -> exact 0 (graph-preserving), not nan
    z = think_distill_loss(talk_logits, think_l, torch.full_like(dl_labels, -100))
    assert float(z) == 0.0
    # weighted KL: uniform weight == plain mean; zero weight -> 0
    kl_u = think_distill_loss(talk_logits, think_l, dl_labels)
    kl_w1 = think_distill_loss(talk_logits, think_l, dl_labels, weight=torch.ones(B, L))
    assert torch.allclose(kl_u, kl_w1, atol=1e-5)
    assert think_distill_loss(talk_logits, think_l, dl_labels, weight=torch.zeros(B, L)).abs() < 1e-6

    # think_entropy_norm in [0,1]; uniform logits -> ~1, peaked -> ~0
    Hn = think_entropy_norm(think_l)
    assert Hn.shape == (B, L) and (Hn >= -1e-6).all() and (Hn <= 1 + 1e-6).all()
    assert think_entropy_norm(torch.zeros(1, 1, V)).item() > 0.99           # uniform -> max entropy
    peaked = torch.full((1, 1, V), -1e4); peaked[0, 0, 0] = 1e4
    assert think_entropy_norm(peaked).item() < 0.01                          # one-hot -> ~0

    # recoverable_mask: gold in top-K but think top-1 wrong
    rl = torch.zeros(1, 3, V); lab = torch.tensor([[5, -100, 7]])
    rl[0, 0, 9] = 10.0; rl[0, 0, 5] = 5.0     # pos0: top1=9 (wrong, gold5 in top-K) -> recoverable
    rl[0, 2, 7] = 10.0                          # pos2: top1=7 == gold -> NOT (think right)
    rm = recoverable_mask(rl, lab, top_k=K)
    assert bool(rm[0, 0]) and not bool(rm[0, 1]) and not bool(rm[0, 2])

    # build_think_next_teacher: shape [B, Ln], finite, reuses one commit decision
    class _StubLM:                                       # logits = inputs_embeds @ W^T
        def __init__(self, Wt): self.Wt = Wt
        def __call__(self, *, inputs_embeds, **kw):
            class O: pass
            o = O(); o.logits = inputs_embeds @ self.Wt; return o
    Ln, Lc = 4, 4                                        # noisy + clean halves; block_size divides Ln
    full_ids = torch.randint(0, V - 1, (B, Ln + Lc))
    full_ids[:, :Ln][torch.rand(B, Ln) < 0.7] = mask_id   # some masked in the noisy half
    s1 = torch.randn(B, Ln + Lc, V)
    stub = _StubLM(W.t())
    for thr in (0.0, 0.99):                              # 0.0 commits the whole prefix; 0.99 ~ none
        s2, am, cm = build_think_next_teacher(stub, W, full_ids.clone(), s1, Ln, mask_id,
                                              block_size=2, threshold=thr)
        assert s2.shape == (B, Ln, V) and torch.isfinite(s2).all()
        assert cm.shape == (B, Ln) and bool((cm <= (full_ids[:, :Ln] == mask_id)).all())  # commits subset of masked
    print("t3d_topk_talk selftest OK")


if __name__ == "__main__":
    _selftest()
