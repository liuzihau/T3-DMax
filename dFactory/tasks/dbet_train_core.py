# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
"""DBet core training step (pure torch — no VeOmni), so it can be unit/smoke-tested off-cluster.

One FROZEN-heavy dual-stream forward -> DMax decode_uniform commit (one pass) -> drafter forward over
[prefix+clean ; noisy] -> decayed CE + confidence BCE on the remaining-masked vs golden. Imported by
`train_dbet.py` (the VeOmni trainer) and by `smoke_dbet.py` (the off-cluster test). `args` is duck-typed:
needs `args.train.block_size`, `args.train.heavy_commit_threshold`, `args.train.conf_loss_weight`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

MASK_ID = 156895


def heavy_commit(noisy_logits, noisy_ids, mask_id, block_size, threshold):
    """DMax decode_uniform commit — ONE pass, argmax, per block, left-to-right prefix of masked positions until
    the first below-threshold (no fallback: training needs no guaranteed progress).
    noisy_logits [B,L,V], noisy_ids [B,L] -> (post_commit_ids [B,L], remaining_mask [B,L] bool)."""
    B, L, _ = noisy_logits.shape
    nb = L // block_size
    probs = torch.softmax(noisy_logits.float(), dim=-1)
    argmax = probs.argmax(dim=-1)                                       # [B,L]
    conf = probs.gather(-1, argmax.unsqueeze(-1)).squeeze(-1)           # [B,L]
    mask = noisy_ids == mask_id
    is_low = (mask & (conf < threshold)).view(B, nb, block_size)
    has_failed = torch.cumsum(is_low.long(), dim=-1) > 0               # any low-conf masked at/before, per block
    commit = (mask.view(B, nb, block_size) & (~has_failed)).view(B, L)
    post = torch.where(commit, argmax, noisy_ids)
    remaining = mask & (~commit)
    return post, remaining


def derive_drafter_mask(dual_mask, L):
    """Drafter mask = noisy-query rows of the dual-stream prototype, columns reordered to [clean ; noisy]
    (matching DbetAttention keys = [prefix_kv(=clean) ; canvas(=noisy)]). dual_mask [B,1,2L,2L] over
    [noisy(0:L) | clean(L:2L)] -> [B,1,L,2L]. Noisy block i attends clean blocks < i (M_OBC) + own noisy
    block (M_BD); clean includes the prompt (early blocks) so the prompt is attended by all."""
    noisy_rows = dual_mask[:, :, :L, :]
    return torch.cat([noisy_rows[:, :, :, L:2 * L], noisy_rows[:, :, :, :L]], dim=-1)


def decay_weights(remaining_mask, block_size, mode="dbet", base=0.9, floor=0.1,
                  head=0.95, tail=0.8, window=6, gamma=14.0):
    """Per-block left-to-right loss-weight over remaining positions. k=0 at the first remaining position in the
    block; weight is 0 outside remaining. `mode` selects the schedule (config.loss_decay_mode):
      - "dbet"          : w[k] = max(base^k, floor)                              (original; base 0.9, floor 0.1)
      - "dbet_twophase" : gentle head^k for k<window, then steeper tail decay -> concentrate weight on the first
                          `window` tokens. w[k<W]=head^k ; w[k>=W]=head^(W-1)*tail^(k-W+1) ; floored at `floor`.
      - "dflash"        : w[k] = exp(-k/gamma)  (DFlash Eq.4 with 0-based k == their 1-based exp(-(k-1)/γ);
                          floored at `floor`). gamma ~ block_size/2 (paper: 7@bs16, 5@bs10, 4@bs8).
    remaining_mask [B,L] -> w [B,L]."""
    B, L = remaining_mask.shape
    nb = L // block_size
    rem = remaining_mask.view(B, nb, block_size)
    k = (torch.cumsum(rem.long(), dim=-1) - 1).clamp(min=0).float()
    if mode == "dflash":
        wk = torch.exp(-k / gamma)
    elif mode == "dbet_twophase":
        boundary = head ** (window - 1)
        wk = torch.where(k < window, head ** k, boundary * tail ** (k - (window - 1)).clamp(min=0.0))
    else:  # "dbet" (default, original)
        wk = base ** k
    w = torch.clamp(wk, min=floor) * rem.float()
    return w.view(B, L)


def decay_kwargs(args):
    """Extract the loss-decay schedule kwargs from args.train (all optional; safe defaults == original 'dbet')."""
    t = args.train
    return dict(
        mode=getattr(t, "loss_decay_mode", "dbet"),
        base=getattr(t, "loss_decay_base", 0.9),
        floor=getattr(t, "loss_decay_floor", 0.1),
        head=getattr(t, "loss_decay_head", 0.95),
        tail=getattr(t, "loss_decay_tail", 0.8),
        window=getattr(t, "loss_decay_window", 6),
        gamma=getattr(t, "loss_decay_gamma", 14.0),
    )


def dbet_forward(core, micro_batch, args, mask_id=MASK_ID, return_post_commit=False, return_heavy_logits=False):
    """Shared FROZEN-heavy -> commit -> drafter forward (no loss). Used by BOTH `dbet_train_step` (with grad on
    the drafter) and the eval pass (wrapped in no_grad) so the two can never drift apart.
    `core` is the UNWRAPPED model; micro_batch carries the dual stream (input_ids=[noisy|clean] [B,2L],
    attention_mask=[B,1,2L,2L] block prototype, position_ids=[B,2L], noisy_input_ids=[B,L]).
    Returns: logits [B,L,V], conf [B,L] (or None), remaining [B,L] bool, clean_ids [B,L] (golden).
    Optional trailing elements (fixed order): heavy_logits [B,L,V] (frozen heavy's noisy dist, for the
    DSpark-style align/accept targets) if return_heavy_logits; then post_commit [B,L] if return_post_commit."""
    cfg = core.config
    bs, thr = args.train.block_size, args.train.heavy_commit_threshold

    full = micro_batch["input_ids"]                                    # [B, 2L] = [noisy | clean]
    attn = micro_batch["attention_mask"]
    pos = micro_batch["position_ids"]
    L = micro_batch["noisy_input_ids"].shape[1]
    noisy_ids, clean_ids = full[:, :L], full[:, L:]                    # clean_ids = golden answer

    # 1) frozen heavy dual-stream forward (its TRAINED layout -> valid hidden); harvest both halves
    with torch.no_grad():
        hout = core.heavy(input_ids=full, attention_mask=attn, position_ids=pos,
                          use_cache=False, output_hidden_states=True, output_router_logits=False, return_dict=True)
    h_sel = torch.cat([hout.hidden_states[i] for i in cfg.sel_layers_list], dim=-1)   # [B,2L,m*D]
    noisy_h_sel, clean_h_sel = h_sel[:, :L], h_sel[:, L:]
    noisy_h_last = hout.hidden_states[-1][:, :L]
    noisy_logits = hout.logits[:, :L]

    # 2) heavy one-pass decode_uniform commit on the noisy logits -> committed / remaining
    post_commit, remaining = heavy_commit(noisy_logits, noisy_ids, mask_id, bs, thr)

    # 3) drafter forward: clean(+prompt) hidden -> prefix KV; noisy -> canvas; mask = [clean ; noisy]
    out = core.draft(
        input_ids=post_commit, heavy_logits=noisy_logits,
        h_sel_denoise=noisy_h_sel, h_last_denoise=noisy_h_last, h_sel_prefix=clean_h_sel,
        attention_mask=derive_drafter_mask(attn, L), position_ids=pos, denoise_mask=None, tau=None,
    )
    ret = [out["logits"], out["conf"], remaining, clean_ids]
    if return_heavy_logits:
        ret.append(noisy_logits)                                       # frozen heavy dist (detached; no_grad above)
    if return_post_commit:
        ret.append(post_commit)
    return tuple(ret)


def first_t_position_acc(correct, remaining, block_size, tpf):
    """Avg-OF-POSITION top-1 accuracy over the first `tpf` remaining positions per block (per-position rate then
    mean over k<tpf) -- the heavy commits ~tpf tokens/forward, so this is the decision-relevant window.
    correct, remaining: [B,L] bool. Returns a python float (nan if no data). Cheap: a `tpf`-iter loop of tensor ops."""
    B, L = remaining.shape
    nb = L // block_size
    k = (torch.cumsum(remaining.view(B, nb, block_size).long(), dim=-1) - 1).clamp(min=0).view(B, L)
    accs = []
    for kk in range(tpf):
        sel = remaining & (k == kk)
        tot = sel.sum()
        if int(tot) > 0:
            accs.append((correct & sel).sum().float() / tot)
    return float(torch.stack(accs).mean()) if accs else float("nan")


def soft_embed(logits_sel, embed_layer, mask_id, tau, top_k):
    """DMax soft-embed: softmax(logits/tau) -> top-k weighted token embeds + residual*embed(MASK), renormalized.
    logits_sel [n,V] -> [n,D]. (Matches generate_dbet._soft_embed / the decode re-feed.)"""
    device = logits_sel.device
    probs = torch.softmax(logits_sel.float() / max(float(tau), 1e-6), dim=-1)
    topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)
    residual = (1.0 - topk_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    topk_emb = embed_layer(topk_idx).float()
    mask_emb = embed_layer(torch.tensor([mask_id], device=device)).float()
    s = (topk_emb * topk_probs.unsqueeze(-1)).sum(dim=1) + mask_emb * residual
    tgt = (topk_emb.norm(dim=-1) * topk_probs).sum(dim=-1, keepdim=True) + mask_emb.norm() * residual
    s = s * (tgt / (s.norm(dim=-1, keepdim=True) + 1e-6))
    return s.to(embed_layer.weight.dtype)


def _heavy_verify_pass(core, micro_batch, heavy_logits1, committed, mask_id, tau, top_k):
    """2nd heavy pass = the VERIFIER. Re-feed the 1st-pass committed positions as SOFT-embeds (the blur lets the
    heavy RECONSIDER a low-confidence commit -- a hard-token re-feed near-copies and finds nothing); remaining
    stay MASK. Forward the frozen heavy; return its logits over the L noisy positions = the settled/verifier
    distribution across the FULL answer region (committed re-predicted + remaining). +1 frozen heavy forward."""
    L = micro_batch["noisy_input_ids"].shape[1]
    full = micro_batch["input_ids"]
    noisy_ids, clean_ids = full[:, :L], full[:, L:]
    embed = core.draft.frozen_embed
    noisy_emb = embed(noisy_ids).clone()                              # golden@revealed, embed(MASK)@masked
    if bool(committed.any()):
        noisy_emb[committed] = soft_embed(heavy_logits1[committed], embed, mask_id, tau, top_k)
    full_emb = torch.cat([noisy_emb, embed(clean_ids)], dim=1)
    with torch.no_grad():
        out2 = core.heavy(inputs_embeds=full_emb, attention_mask=micro_batch["attention_mask"],
                          position_ids=micro_batch["position_ids"], use_cache=False,
                          output_router_logits=False, return_dict=True)
    return out2.logits[:, :L]


def dbet_train_step(model, micro_batch, n_micro_batches, args, mask_id=MASK_ID, return_metrics=False):
    """DBet core step (requires the dual stream already in micro_batch: input_ids=[noisy|clean] [B,2L],
    attention_mask=[B,1,2L,2L] block-diffusion prototype, position_ids=[B,2L], noisy_input_ids=[B,L]).
    Returns loss/n_micro_batches (and a metrics dict if return_metrics).
    Target = the heavy's 2nd (VERIFIER) pass, ALWAYS: the drafter's input is the POST-commit state, so the only
    coherent target is the heavy's next-step distribution (commits re-fed as soft embeds) over the FULL answer
    region -- committed re-predicted at peak weight (FIX supervision) + remaining decayed (EXTEND supervision).
    (The old `align_to_2nd_pass=False` route targeted the PRE-commit 1st-pass logits on remaining-only: an
    input/target mismatch with zero FIX signal. Removed 2026-07-09; the config flag now only guards startup.)"""
    core = model.module if hasattr(model, "module") else model         # unwrap FSDP1 if present
    bs = args.train.block_size

    logits, conf, remaining, clean_ids, heavy_logits, post_commit = dbet_forward(
        core, micro_batch, args, mask_id, return_heavy_logits=True, return_post_commit=True)
    rem_orig = remaining.bool()
    masked_full = (micro_batch["noisy_input_ids"] == mask_id)
    committed_ctx = masked_full & (~rem_orig)                          # ALL heavy commits (context for the verify pass)
    # --- pad-tail gate: supervise only the answer + ~32 trailing EOS (the data's `labels` truncation), else the
    #     loss/acc drown in the ~1000 trivial trailing-pad positions (EOS == pad here). sup subset of masked;
    #     falls back to all-masked if labels absent.
    labels = micro_batch.get("labels")
    sup = (labels != -100) if labels is not None else masked_full
    rem = rem_orig & sup
    w = decay_weights(remaining, bs, **decay_kwargs(args)) * sup.float()   # decay (block-position correct), zeroed off-sup

    # FULL supervised region = committed | remaining; target = 2nd heavy (verifier) pass; committed = peak weight.
    tau = float(getattr(args.train, "heavy_soft_tau", 1.0))
    topk = int(getattr(args.train, "heavy_soft_top_k", 1))
    target_logits = _heavy_verify_pass(core, micro_batch, heavy_logits, committed_ctx, mask_id, tau, topk)
    committed = committed_ctx & sup                                   # supervised commits (for loss + fix metrics)
    region = sup                                                      # supervised masked region (= committed | rem)
    w[committed] = 1.0                                                # committed -> first-masked-index weight (no decay)

    wl = w[region]                                                    # [n]
    denom = wl.sum().clamp_min(1.0)

    # --- Alignment: match the (frozen, detached) heavy target. DSpark: acceptance depends on distribution OVERLAP
    #     (1 - 0.5*L1), so L1/TV is the direct objective; CE-to-target-argmax stabilizes; small golden nudge.
    ce_a  = float(getattr(args.train, "align_ce_weight", 0.1))
    l1_a  = float(getattr(args.train, "align_l1_weight", 1.0))
    gld_a = float(getattr(args.train, "golden_ce_weight", 0.1))
    dlog = logits[region].float()                                     # [n,V] draft logits on the region
    tlog = target_logits[region].float()                             # [n,V] heavy target logits (1st or 2nd pass)
    targ = tlog.argmax(-1)                                            # [n]
    ce_heavy  = F.cross_entropy(dlog, targ,              reduction="none")   # [n] match heavy top token
    ce_golden = F.cross_entropy(dlog, clean_ids[region], reduction="none")  # [n] small nudge to golden
    l1 = (torch.softmax(dlog, -1) - torch.softmax(tlog, -1)).abs().sum(-1)   # [n] L1; TV = 0.5*L1
    tok_loss = ((ce_a * ce_heavy + l1_a * l1 + gld_a * ce_golden) * wl).sum() / denom
    loss = tok_loss

    # --- Confidence head regresses the per-position ACCEPTANCE RATE a = 1 - 0.5*L1(draft, heavy target) (DSpark).
    accept_rate = (1.0 - 0.5 * l1).clamp(0.0, 1.0).detach()          # [n]
    conf_loss = None
    if conf is not None:
        c = conf[region].float().clamp(1e-5, 1 - 1e-5)              # [n] drafter conf is a probability
        bce = -(accept_rate * c.log() + (1 - accept_rate) * (1 - c).log())
        conf_loss = (bce * wl).sum() / denom
        loss = loss + args.train.conf_loss_weight * conf_loss
    loss = loss / n_micro_batches
    if not return_metrics:
        return loss                                                   # skip the metric .item() syncs (gated by caller)

    # --- metrics (sync points; the caller only requests these every log_steps) ---
    tpf = int(getattr(args.train, "eval_tpf", 6))
    darg = logits.argmax(-1)
    acc_gold = (darg == clean_ids) & region                           # drafter vs GOLDEN over the (supervised) region
    pad_id = int(getattr(core.config, "pad_token_id", 156892))         # EOS == pad here
    eos_m = (clean_ids == pad_id) & region                            # the KEPT trailing-EOS positions (termination)
    metrics = {
        "tok": float(tok_loss.detach()),
        "ce_heavy": float((ce_heavy * wl).sum() / denom),
        "l1": float((l1 * wl).sum() / denom),
        "acc": float((acc_gold.float() * w).sum() / denom),           # drafter vs golden
        "acc_heavy": float(((dlog.argmax(-1) == targ).float() * wl).sum() / denom),   # drafter vs heavy target
        "accept_rate": float((accept_rate * wl).sum() / denom),       # mean predicted acceptance
        "acc6": first_t_position_acc(acc_gold, rem, bs, tpf),
        "eos_acc": float(((darg == clean_ids) & eos_m).float().sum() / eos_m.float().sum().clamp_min(1)),  # termination
        "n_remaining": int(rem.sum()),
    }
    if bool(committed.any()):
        h2c = target_logits[committed].argmax(-1)                     # 2nd-pass call on committed positions
        commit_tok = post_commit[committed]                          # what the 1st pass committed
        draftc = logits[committed].argmax(-1)
        disagree = (h2c != commit_tok)                               # 2nd pass "finds a 1st-pass mistake"
        nd = int(disagree.sum())
        metrics["n_committed"] = int(committed.sum())
        metrics["commit_wrong_rate"] = float(disagree.float().mean())            # frac of commits the verifier changes
        metrics["fix_acc"] = float((draftc[disagree] == h2c[disagree]).float().mean()) if nd > 0 else float("nan")
    if conf_loss is not None:
        metrics["conf"] = float(conf_loss.detach())
    return loss, metrics                                              # loss already /n_micro_batches above
