# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
"""DBet TRAIN-V2 core step (pure torch — no VeOmni; mockable off-cluster). Design: experiment.md §train-v2.

"Careful teacher, reckless student, and a repair class" — per micro-batch (heavy FROZEN, drafter trains;
the noisy stream arrives ALL-MASK in the answer region: yaml noise_range 1.0/1.0):

  1. G ROLLOUT (careful reference): `g_passes` sequential heavy passes @ `g_threshold` (0.9) with SOFT
     re-feed of commits between passes (mirrors the real decode; hard re-feed would self-copy). Record the
     LAST pass logits **p** + the TRUST WINDOW = committed-after-last-pass (a left-to-right prefix per
     block). Forced progress: a block that commits nothing takes its leftmost masked slot.
  2. A PASS (reckless stage): ONE heavy pass on the all-MASK stream, commit @ thr_A sampled per micro-batch
     from `a_thresholds` {0.3,0.5,0.7} -> a long committed prefix whose tail holds wrong tokens = dense FIX
     material. Its logits/h_sel/h_last feed the drafter (inference-matched conditioning).
  3. DRAFTER, TWO ROUTES on the post-A state (identical soft-embed + h_sel conditioning):
       route H — commits shown HARD (deploy default);  route M — committed region shown as MASK
     (can't copy -> must re-derive; un-anchoring scales with the heavy's own uncertainty via the soft-embed).
  4. VERIFIER: one heavy pass on the post-A state (commits soft re-fed) -> **p'** (next-step distribution).
  5. ROUTE-SPLIT LOSS (block decay + committed peak weight + conf-head accept regression vs p', both routes):
       H: 0.1*CE(q,argmax p') + 1.0*L1(q,p') + 0.1*CE(q,golden-text)
       M: 0.3*CE(q,argmax p)[TRUST WINDOW ONLY] + 0.7*L1(q,p') + 0.1*CE(q,golden-text)
  6. ON-POLICY ROUND: commit route-H's (detached) output like deployment — EXTEND = per-block left-to-right
     prefix of remaining while conf >= 0.8 (first forced), FIX = committed & conf >= 0.9 & disagrees — then
     one heavy pass on the drafter-modified state (drafter commits soft-embedded from q_H top-k 2, heavy
     commits from p_A), re-commit @ thr_A WITH revision (the heavy accepts/rejects the drafter's work), and
     repeat 3–5 on that state against the SAME p / trust window. loss = mean of the 4 route-instances.

Cost: 9 frozen heavy forwards (5 G + A + verify1 + round2 + verify2) + 4 drafter forwards per step.
`args` duck-typed: args.train needs block_size, conf_loss_weight, align_ce_weight, align_l1_weight,
golden_ce_weight, heavy_soft_tau, heavy_soft_top_k, loss_decay_*, and the v2 knobs (g_passes, g_threshold,
a_thresholds, m_ce_p_weight, m_l1_weight, sendback_*).
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from dbet_train_core import (
    MASK_ID,
    decay_kwargs,
    decay_weights,
    derive_drafter_mask,
    first_t_position_acc,
    soft_embed,
)


# ============================================================================
#                     DMax commit (with revision) + G rollout
# ============================================================================
def dmax_commit_revise(logits, ids, committed, mask_id, block_size, threshold, force_progress=False):
    """One DMax decode_uniform-style commit step WITH revision, mirroring inference: previously-committed
    slots take the FRESH argmax (they may flip — decode_uniform rewrites all active&committed slots every
    pass), then the left-to-right prefix of still-masked slots commits while conf >= threshold (stop at the
    first below-threshold; no fallback unless `force_progress`, which commits the leftmost masked slot of any
    block that would otherwise commit nothing — the G rollout needs guaranteed progress for its trust window).
    logits [B,L,V], ids [B,L], committed [B,L] bool -> (new_ids, new_committed, remaining)."""
    B, L, _ = logits.shape
    nb = L // block_size
    probs = torch.softmax(logits.float(), dim=-1)
    argmax = probs.argmax(dim=-1)                                       # [B,L]
    conf = probs.gather(-1, argmax.unsqueeze(-1)).squeeze(-1)           # [B,L]

    ids = torch.where(committed, argmax, ids)                           # revision of prior commits
    mask = ids == mask_id
    is_low = (mask & (conf < threshold)).view(B, nb, block_size)
    has_failed = torch.cumsum(is_low.long(), dim=-1) > 0               # any low-conf masked at/before, per block
    newly = (mask.view(B, nb, block_size) & (~has_failed))              # [B,nb,bs]
    if force_progress:
        need = mask.view(B, nb, block_size).any(-1) & (~newly.any(-1))  # blocks with masks but no commit
        if bool(need.any()):
            leftmost = mask.view(B, nb, block_size).long().argmax(-1)   # first masked index per block
            forced = torch.zeros_like(newly)
            forced.scatter_(-1, leftmost.unsqueeze(-1), True)
            newly = newly | (forced & need.unsqueeze(-1))
    newly = newly.view(B, L)
    ids = torch.where(newly, argmax, ids)
    committed = committed | newly
    remaining = mask & (~newly)
    return ids, committed, remaining


def _dual_forward(core, micro_batch, noisy_emb=None, noisy_ids=None, want_hidden=False):
    """One frozen dual-stream heavy forward. Feed either noisy token ids (pass 1 / hard) or noisy embeds
    (soft re-feed passes); the clean half always rides along as ids->embeds. Returns the raw heavy output."""
    L = micro_batch["noisy_input_ids"].shape[1]
    clean_ids = micro_batch["input_ids"][:, L:]
    kw = dict(attention_mask=micro_batch["attention_mask"], position_ids=micro_batch["position_ids"],
              use_cache=False, output_router_logits=False, return_dict=True,
              output_hidden_states=want_hidden)
    embed = core.draft.frozen_embed
    with torch.no_grad():
        if noisy_emb is not None:
            return core.heavy(inputs_embeds=torch.cat([noisy_emb, embed(clean_ids)], dim=1), **kw)
        return core.heavy(input_ids=torch.cat([noisy_ids, clean_ids], dim=1), **kw)


def g_rollout(core, micro_batch, mask_id, g_passes, g_threshold, block_size, tau, top_k):
    """The careful reference: from the all-MASK noisy stream, `g_passes` heavy passes @ g_threshold with soft
    re-feed of commits between passes. Returns (p_logits [B,L,V] = the LAST executed pass's distribution,
    trust [B,L] bool = committed after its commit — the per-block left-to-right prefix the reference has
    actually settled; p is only a trustworthy target inside it). Stops early if every block settles."""
    L = micro_batch["noisy_input_ids"].shape[1]
    embed = core.draft.frozen_embed
    ids = micro_batch["noisy_input_ids"].clone()
    committed = torch.zeros_like(ids, dtype=torch.bool)
    logits = None
    for t in range(g_passes):
        if t == 0:
            out = _dual_forward(core, micro_batch, noisy_ids=ids)
        else:
            noisy_emb = embed(ids).clone()
            noisy_emb[committed] = soft_embed(logits[committed], embed, mask_id, tau, top_k)
            out = _dual_forward(core, micro_batch, noisy_emb=noisy_emb)
        logits = out.logits[:, :L]
        ids, committed, remaining = dmax_commit_revise(
            logits, ids, committed, mask_id, block_size, g_threshold, force_progress=True)
        if not bool(remaining.any()):
            break
    return logits, committed


def verify_pass(core, micro_batch, ids, committed, logits_src, mask_id, tau, top_k):
    """The VERIFIER: re-forward the post-commit state with committed slots soft re-fed from `logits_src`
    (the blur lets the heavy RECONSIDER; hard re-feed near-copies) -> p' over the L noisy positions."""
    L = micro_batch["noisy_input_ids"].shape[1]
    embed = core.draft.frozen_embed
    noisy_emb = embed(ids).clone()
    if bool(committed.any()):
        noisy_emb[committed] = soft_embed(logits_src[committed], embed, mask_id, tau, top_k)
    out = _dual_forward(core, micro_batch, noisy_emb=noisy_emb)
    return out.logits[:, :L]


# ============================================================================
#                     drafter routes + sendback + route loss
# ============================================================================
def drafter_forward(core, micro_batch, ids, heavy_logits, h_sel, h_last, clean_h_sel):
    """One drafter forward on canvas `ids` (route H passes post-commit ids; route M passes them with the
    committed region re-MASKed). Conditioning identical across routes."""
    L = micro_batch["noisy_input_ids"].shape[1]
    return core.draft(
        input_ids=ids, heavy_logits=heavy_logits,
        h_sel_denoise=h_sel, h_last_denoise=h_last, h_sel_prefix=clean_h_sel,
        attention_mask=derive_drafter_mask(micro_batch["attention_mask"], L),
        position_ids=micro_batch["position_ids"], denoise_mask=None, tau=None,
    )


def sendback_commit(q_logits, q_conf, ids, committed, remaining, mask_id, block_size, ext_thr, fix_thr):
    """Deployment-faithful drafter commit (mirrors generate_dbet): EXTEND = per-block left-to-right prefix of
    the remaining masked slots while conf >= ext_thr, first remaining slot ALWAYS committed (the keep[0]=True
    progress rule); FIX = committed slot with conf >= fix_thr whose draft argmax disagrees.
    Returns (new_ids, new_committed, draft_slots [B,L] = every slot the drafter wrote)."""
    B, L = ids.shape
    nb = L // block_size
    darg = q_logits.argmax(-1)                                          # [B,L]
    rem = remaining.view(B, nb, block_size)
    k = (torch.cumsum(rem.long(), dim=-1) - 1).clamp(min=0)             # index among remaining, per block
    ok = (q_conf >= ext_thr).view(B, nb, block_size)
    fail = rem & (~ok)
    BIG = block_size + 1
    kfail = torch.where(fail, k, torch.full_like(k, BIG)).amin(dim=-1)  # first failing remaining-index
    cut = kfail.clamp(min=1)                                            # forced first commit
    extend = (rem & (k < cut.unsqueeze(-1))).view(B, L)
    fix = committed & (q_conf >= fix_thr) & (darg != ids)
    draft_slots = extend | fix
    new_ids = torch.where(draft_slots, darg, ids)
    return new_ids, committed | extend, draft_slots


def _route_loss(logits, conf, pp_logits, p_logits, trust, clean_ids, region, w, args,
                ce_pp_w, ce_p_w, l1_w):
    """One route-instance loss vs the verifier p' (+ optional trust-gated CE toward the careful reference p).
    Same family as v1: CE-to-target-argmax + L1/TV + small golden-text CE, decay-weighted; conf head does BCE
    against the acceptance rate 1 - 0.5*L1(q, p'). Returns (loss, parts) — parts carry detached diagnostics."""
    gld_w = float(getattr(args.train, "golden_ce_weight", 0.1))
    conf_w = float(getattr(args.train, "conf_loss_weight", 1.0))
    wl = w[region]
    denom = wl.sum().clamp_min(1.0)
    dlog = logits[region].float()
    tlog = pp_logits[region].float()
    l1 = (torch.softmax(dlog, -1) - torch.softmax(tlog, -1)).abs().sum(-1)
    ce_pp = F.cross_entropy(dlog, tlog.argmax(-1), reduction="none")
    ce_gold = F.cross_entropy(dlog, clean_ids[region], reduction="none")
    tok = l1_w * l1 + ce_pp_w * ce_pp + gld_w * ce_gold
    if ce_p_w > 0:
        ce_p = F.cross_entropy(dlog, p_logits[region].float().argmax(-1), reduction="none")
        tok = tok + ce_p_w * ce_p * trust[region].float()               # careful teacher: trust window only
    tok_loss = (tok * wl).sum() / denom
    loss = tok_loss
    accept = (1.0 - 0.5 * l1).clamp(0.0, 1.0).detach()
    conf_loss = None
    if conf is not None:
        c = conf[region].float().clamp(1e-5, 1 - 1e-5)
        bce = -(accept * c.log() + (1 - accept) * (1 - c).log())
        conf_loss = (bce * wl).sum() / denom
        loss = loss + conf_w * conf_loss
    parts = {"tok": tok_loss.detach(), "l1": (l1 * wl).sum().detach() / denom,
             "accept": (accept * wl).sum() / denom,
             "conf": conf_loss.detach() if conf_loss is not None else None}
    return loss, parts


def _fix_metrics(q_logits, pp_logits, ids, committed):
    """FIX diagnostics on committed slots: how often the verifier flips a commit, and how often the drafter's
    argmax names the verifier's correction on exactly those slots."""
    if not bool(committed.any()):
        return float("nan"), float("nan"), 0
    v = pp_logits[committed].argmax(-1)
    tok = ids[committed]
    d = q_logits[committed].argmax(-1)
    dis = v != tok
    nd = int(dis.sum())
    wrong_rate = float(dis.float().mean())
    fix_acc = float((d[dis] == v[dis]).float().mean()) if nd > 0 else float("nan")
    return wrong_rate, fix_acc, nd


# ============================================================================
#                                the v2 step
# ============================================================================
def _one_round(core, micro_batch, args, mask_id, ids, committed, remaining, heavy_logits,
               h_sel, h_last, clean_h_sel, p_logits, trust_sup, sup, clean_ids, tag, metrics, want_metrics):
    """Rounds 1 and 2 share this: verifier -> two drafter routes -> route-split losses (+ metrics under
    `tag` suffix). Returns (loss_H + loss_M, route-H output dict for the sendback)."""
    t = args.train
    bs = t.block_size
    tau = float(getattr(t, "heavy_soft_tau", 1.0))
    topk = int(getattr(t, "heavy_soft_top_k", 1))
    pp = verify_pass(core, micro_batch, ids, committed, heavy_logits, mask_id, tau, topk)

    w = decay_weights(remaining, bs, **decay_kwargs(args)) * sup.float()
    committed_sup = committed & sup
    w[committed_sup] = 1.0                                              # committed -> peak weight, no decay
    region = sup

    ids_m = torch.where(committed, torch.full_like(ids, mask_id), ids)   # route M: hide the commits
    outH = drafter_forward(core, micro_batch, ids, heavy_logits, h_sel, h_last, clean_h_sel)
    outM = drafter_forward(core, micro_batch, ids_m, heavy_logits, h_sel, h_last, clean_h_sel)

    lossH, pH = _route_loss(outH["logits"], outH["conf"], pp, p_logits, trust_sup, clean_ids, region, w, args,
                            ce_pp_w=float(getattr(t, "align_ce_weight", 0.1)), ce_p_w=0.0,
                            l1_w=float(getattr(t, "align_l1_weight", 1.0)))
    lossM, pM = _route_loss(outM["logits"], outM["conf"], pp, p_logits, trust_sup, clean_ids, region, w, args,
                            ce_pp_w=0.0, ce_p_w=float(getattr(t, "m_ce_p_weight", 0.3)),
                            l1_w=float(getattr(t, "m_l1_weight", 0.7)))
    if want_metrics:
        for name, out, parts in (("H", outH, pH), ("M", outM, pM)):
            metrics[f"tok_{name}{tag}"] = float(parts["tok"])
            metrics[f"l1_{name}{tag}"] = float(parts["l1"])
            if parts["conf"] is not None:
                metrics[f"conf_{name}{tag}"] = float(parts["conf"])
            acc_gold = (out["logits"].argmax(-1) == clean_ids) & region
            metrics[f"acc_{name}{tag}"] = float((acc_gold.float() * w).sum() / w[region].sum().clamp_min(1.0))
            wr, fa, nd = _fix_metrics(out["logits"], pp, ids, committed_sup)
            metrics[f"fix_acc_{name}{tag}"] = fa
            if name == "H":
                metrics[f"commit_wrong_rate{tag}"] = wr
                metrics[f"n_committed{tag}"] = int(committed_sup.sum())
                metrics[f"acc6_{name}{tag}"] = first_t_position_acc(acc_gold, remaining & sup, bs,
                                                                    int(getattr(t, "eval_tpf", 6)))
        # route M vs the careful reference inside the trust window
        if bool(trust_sup.any()):
            mp = (outM["logits"].argmax(-1) == p_logits.argmax(-1)) & trust_sup
            metrics[f"acc_p_M{tag}"] = float(mp.float().sum() / trust_sup.float().sum())
    return lossH + lossM, outH


def dbet_v2_train_step(model, micro_batch, n_micro_batches, args, mask_id=MASK_ID, return_metrics=False):
    """TRAIN-V2 step (drop-in signature for train_dbet's loop; see module docstring for the recipe).
    Requires the dual stream in micro_batch with the noisy answer region ALL-MASK (yaml noise_range 1.0/1.0).
    Returns loss/n_micro_batches (and a metrics dict if return_metrics)."""
    core = model.module if hasattr(model, "module") else model
    t = args.train
    bs = t.block_size
    tau = float(getattr(t, "heavy_soft_tau", 1.0))
    topk = int(getattr(t, "heavy_soft_top_k", 1))
    cfg = core.config
    L = micro_batch["noisy_input_ids"].shape[1]
    full = micro_batch["input_ids"]
    noisy_ids, clean_ids = full[:, :L], full[:, L:]
    labels = micro_batch.get("labels")
    masked_full = noisy_ids == mask_id
    sup = (labels != -100) if labels is not None else masked_full

    # 1) careful reference p + trust window (5 careful passes, no hidden states needed)
    p_logits, trust = g_rollout(core, micro_batch, mask_id,
                                int(getattr(t, "g_passes", 5)), float(getattr(t, "g_threshold", 0.9)),
                                bs, tau, topk)
    trust_sup = trust & sup

    # 2) reckless A pass: full-featured forward (logits + h_sel/h_last feed the drafter), sampled threshold
    a_choices = [float(x) for x in str(getattr(t, "a_thresholds", "0.3,0.5,0.7")).split(",") if x.strip()]
    thr_a = random.choice(a_choices)
    outA = _dual_forward(core, micro_batch, noisy_ids=noisy_ids, want_hidden=True)
    h_sel_all = torch.cat([outA.hidden_states[i] for i in cfg.sel_layers_list], dim=-1)
    h_sel, clean_h_sel = h_sel_all[:, :L], h_sel_all[:, L:]
    h_last = outA.hidden_states[-1][:, :L]
    logitsA = outA.logits[:, :L]
    idsA, committedA, remainingA = dmax_commit_revise(
        logitsA, noisy_ids, torch.zeros_like(masked_full), mask_id, bs, thr_a)

    metrics = {"thr_a": thr_a}
    if return_metrics and bool(sup.any()):
        metrics["trust_frac"] = float((trust_sup.float().sum() / sup.float().sum()))

    # 3-5) round 1 (off-policy: heavy-made state)
    loss_r1, outH = _one_round(core, micro_batch, args, mask_id, idsA, committedA, remainingA, logitsA,
                               h_sel, h_last, clean_h_sel, p_logits, trust_sup, sup, clean_ids,
                               "", metrics, return_metrics)

    # 6) sendback: route-H acts like deployment (detached — no grad through the frozen heavy anyway)
    ids2, committed2, draft_slots = sendback_commit(
        outH["logits"].detach(), (outH["conf"].detach() if outH["conf"] is not None
                                  else torch.zeros_like(idsA, dtype=torch.float)),
        idsA, committedA, remainingA, mask_id, bs,
        float(getattr(t, "sendback_extend_threshold", 0.8)), float(getattr(t, "sendback_fix_threshold", 0.9)))

    # round-2 heavy pass on the drafter-modified state: heavy commits soft re-fed from p_A, drafter commits
    # soft-embedded from q_H (top-k 2) — mirrors the inference embeddings re-feed
    embed = core.draft.frozen_embed
    noisy_emb2 = embed(ids2).clone()
    heavy_slots2 = committed2 & (~draft_slots)
    if bool(heavy_slots2.any()):
        noisy_emb2[heavy_slots2] = soft_embed(logitsA[heavy_slots2], embed, mask_id, tau, topk)
    if bool(draft_slots.any()):
        noisy_emb2[draft_slots] = soft_embed(outH["logits"].detach()[draft_slots], embed, mask_id,
                                             float(getattr(t, "sendback_tau", 1.0)),
                                             int(getattr(t, "sendback_top_k", 2)))
    outB = _dual_forward(core, micro_batch, noisy_emb=noisy_emb2, want_hidden=True)
    h_sel_allB = torch.cat([outB.hidden_states[i] for i in cfg.sel_layers_list], dim=-1)
    h_selB = h_sel_allB[:, :L]
    h_lastB = outB.hidden_states[-1][:, :L]
    logitsB = outB.logits[:, :L]
    ids3, committed3, remaining3 = dmax_commit_revise(logitsB, ids2, committed2, mask_id, bs, thr_a)

    if return_metrics and bool(draft_slots.any()):
        metrics["n_draft_committed"] = int(draft_slots.sum())
        metrics["draft_commit_survive"] = float((ids3[draft_slots] == ids2[draft_slots]).float().mean())

    # 3-5 again) round 2 (on-policy: the state carries the drafter's own commits)
    loss_r2, _ = _one_round(core, micro_batch, args, mask_id, ids3, committed3, remaining3, logitsB,
                            h_selB, h_lastB, clean_h_sel, p_logits, trust_sup, sup, clean_ids,
                            "_r2", metrics, return_metrics)

    loss = 0.25 * (loss_r1 + loss_r2)                                   # mean of the 4 route-instances
    loss = loss / n_micro_batches
    if not return_metrics:
        return loss
    metrics["loss_v2"] = float(loss.detach()) * n_micro_batches
    return loss, metrics
