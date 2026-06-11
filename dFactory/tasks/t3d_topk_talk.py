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
    from .t3d_topk_soft_embed import build_topk_soft_embeds         # package import
except ImportError:                                                  # standalone `python t3d_topk_talk.py`
    from t3d_topk_soft_embed import build_topk_soft_embeds


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


def predict_loss(talk_logits: torch.Tensor, labels: torch.Tensor,
                 predict_mask: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Plain CE on the predict positions (the still-masked ~75%). `predict_mask` is
    bool [B, L]; positions where labels==ignore_index are skipped automatically."""
    lbl = labels.clone()
    lbl[~predict_mask] = ignore_index
    V = talk_logits.shape[-1]
    return F.cross_entropy(talk_logits.view(-1, V), lbl.view(-1), ignore_index=ignore_index)


def think_distill_loss(talk_logits: torch.Tensor, think_logits: torch.Tensor,
                       labels: torch.Tensor, *, temperature: float = 1.0,
                       ignore_index: int = -100, eps: float = 1e-8) -> torch.Tensor:
    """Path-A think->talk distillation: pull the talk's distribution toward the FROZEN
    think's distribution at the predict positions (labels != ignore_index, i.e. the still-
    masked ~75%). Token-decomposed FORWARD KL  D_KL(think || talk)  (== soft cross-entropy
    up to think's entropy, same gradient) -> MASS-COVERING: the talk is pulled to cover
    think's full top-K, not just its top-1. That is the right signal for the Stage-1 cold
    start (reverse KL would be mode-seeking -> collapse onto think's top-1 = the ceiling).

    `temperature` softens both sides (classic distillation T); the loss is scaled by T^2 to
    keep the gradient magnitude comparable to the gold CE term. Returns a scalar = mean
    forward-KL over the predict positions (0 at a perfect match -> clean monitor).
    Computed in fp32. think_logits must be aligned to talk_logits' positions (no shift)."""
    predict = (labels != ignore_index)                        # [B, L] bool
    if not bool(predict.any()):
        return talk_logits.sum() * 0.0                        # keep graph, zero contribution
    T = float(temperature)
    th = (think_logits[predict].float() / T)                  # [N, V]
    tl = (talk_logits[predict].float() / T)                   # [N, V]
    teacher = F.softmax(th, dim=-1)                           # think's soft target
    student_logp = F.log_softmax(tl, dim=-1)
    teacher_logp = F.log_softmax(th, dim=-1)
    kl = (teacher * (teacher_logp - student_logp)).sum(-1)    # [N] forward KL per position
    return kl.mean() * (T * T)


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


def topk_talk_train_step(think, talk, embedding, mask_id, noisy_input_ids, labels,
                         attention_mask, position_ids, flag, *, top_k: int = 10):
    """One anchor-free top-K training step's forward+loss. THE reusable core of the
    training loop (Block 2): the caller (a fork of train_t3_dmax_bd_oput.py) supplies
    a batch + the block-causal `attention_mask` and calls this per micro-batch.

    think: frozen full LLaDA2-Moe (ForCausalLM); talk: trainable 10-layer LLaDA2-Moe.
    `labels` already carries -100 at non-predict positions (the data transform sets
    that), so CE(ignore_index=-100) scores exactly the masked predict positions.
    `flag` (per micro-batch): True → Path B (think's top-K input); False → Path A ([MASK]).
    """
    with torch.no_grad():                                       # think frozen
        think_logits = think(inputs_embeds=embedding(noisy_input_ids),
                             attention_mask=attention_mask, position_ids=position_ids,
                             use_cache=False, return_dict=True).logits
    mode = "topk_soft" if bool(flag) else "mask"
    talk_embeds = build_talk_inputs_embeds(noisy_input_ids, think_logits, embedding, mask_id,
                                           mode=mode, top_k=top_k, keep_mask_residual=False)
    talk_logits = talk(inputs_embeds=talk_embeds, attention_mask=attention_mask,
                       position_ids=position_ids, use_cache=False, return_dict=True).logits
    V = talk_logits.shape[-1]
    return F.cross_entropy(talk_logits.view(-1, V).float(), labels.view(-1), ignore_index=-100)


# ---- reference training step (pseudocode; uses the model parts directly) -----
# Wire this into a fork of train_t3_dmax_bd_oput.py's main loop. Think is frozen and
# called under no_grad; the talk runs ANCHOR-FREE on the top-K-injected embeds.
#
#   think = model.think_model; talk = model.talk_model; head = model.lm_head
#   emb   = model.get_input_embeddings()
#
#   with torch.no_grad():                                   # think frozen, no grad
#       th = think(input_ids=full_ids, attention_mask=block_mask,
#                  output_hidden_states=True, use_cache=False)
#       think_hidden = th.hidden_states[-1]                 # = old "anchor"
#       think_logits = head(think_hidden)                   # [B, L, V]
#
#   # reveal ~25% from the trajectory's first-step commits (data) -> `noisy`, `labels`
#   # (labels[revealed] = -100 to avoid the copy-through-embedding identity leak)
#
#   mode = "topk_soft" if rollout_flag else "mask"          # Path B vs Path A
#   talk_embeds = build_talk_inputs_embeds(noisy, think_logits, emb, mask_id,
#                     mode=mode, top_k=args.t3_topk, keep_mask_residual=False)
#
#   talk_hidden = talk(inputs_embeds=talk_embeds, anchor=None,   # ANCHOR-FREE
#                      attention_mask=block_mask, position_ids=pos, use_cache=False)
#   talk_logits = head(talk_hidden)
#   loss = predict_loss(talk_logits, labels, predict_mask=still_masked)


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

    # predict_loss only counts predict positions
    talk_logits = torch.randn(B, L, V)
    labels = torch.randint(0, V - 1, (B, L))
    loss = predict_loss(talk_logits, labels, predict_mask=masked)
    assert torch.isfinite(loss) and loss > 0
    # zero predict positions -> nan/ignored (cross_entropy over empty -> nan); guard expectation

    # think_distill_loss: forward KL at predict positions (labels!=-100)
    dl_labels = labels.clone(); dl_labels[~masked] = -100
    think_l = torch.randn(B, L, V)
    kl = think_distill_loss(talk_logits, think_l, dl_labels)
    assert torch.isfinite(kl) and kl > 0
    # KL(think||think) == 0 (perfect match), and temperature scaling stays finite
    assert think_distill_loss(think_l, think_l, dl_labels).abs() < 1e-5
    assert torch.isfinite(think_distill_loss(talk_logits, think_l, dl_labels, temperature=2.0))
    # no predict positions -> exact 0 (graph-preserving), not nan
    z = think_distill_loss(talk_logits, think_l, torch.full_like(dl_labels, -100))
    assert float(z) == 0.0
    print("t3d_topk_talk selftest OK")


if __name__ == "__main__":
    _selftest()
