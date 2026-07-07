# Copyright 2026 University of Sydney. Apache-2.0.
"""G2 (SGLang x DBet, Option A) — the DBet block-diffusion decode driving the SGLang heavy + the eager drafter.

Mirrors `generate_dbet.decode_block_dbet` (the no-cache DBet path) exactly, with two swaps:
  - heavy forward: `ModelRunner.forward(inputs_embeds=..., attention_mask=<bool block-causal>, use_cache=False)`
    over [0,be); `logits = out.logits[:,bs:be]`, and `h_sel`/`h_last` come from a `HeavyFeatureTap` (tap.pop()).
  - drafter: the standalone `DbetDraftStack` (load_drafter_standalone), called on the tapped features.
No KV cache (batching, not caching, is the SGLang speedup). The commit rule, soft-embed re-feed, EXTEND/FIX and
exit rule are identical to eager DBet, so accuracy is directly comparable. bf16 (sglang kernels are bf16-only).
"""
import torch

from dinfer.decoding.generate_t3d import dmax_commit_uniform
from dinfer.decoding.generate_dbet import _soft_embed, DbetGenerateStats, _now, MASK_ID, EOS_ID, PAD_ID


def _block_causal_bool(be, block_length, device):
    """Bool block-causal mask [1, be, be] for the SGLang heavy (SDPA bool mask: True = attend)."""
    idx = torch.arange(be, device=device)
    qb = (idx // block_length).unsqueeze(1); kb = (idx // block_length).unsqueeze(0)
    return (kb <= qb).unsqueeze(0)


@torch.no_grad()
def decode_block_dbet_sglang(runner, tap, draft, embed, x, bs, be, attn_bool, heavy_threshold, draft_threshold,
                             max_iters, tau, stats, heavy_tau=1.0, heavy_top_k=1, draft_tau=1.0, draft_top_k=1,
                             draft_committed_soft=False, draft_fix=True):
    """One block, DMax-faithful (== decode_block_dbet): HEAVY (sglang) forward -> dmax_commit_uniform -> soft-embed
    re-feed -> DMax exit; between heavy passes the DRAFTER (on tapped h_sel/h_last) EXTENDs + FIXes."""
    device = x.device
    pos = torch.arange(be, device=device).unsqueeze(0)
    active = (x[0:1, bs:be] == MASK_ID)                                  # original decode region (all-mask at start)
    prefix_embeds = embed(x[:, :bs])                                     # [1, bs, D]
    block_embeds = embed(x[:, bs:be]).clone()                           # [1, blk, D]
    block_logits = None

    it = 0
    while it < max_iters and bool((x[0:1, bs:be] == MASK_ID).any()):
        inputs_embeds = torch.cat([prefix_embeds, block_embeds], dim=1)   # [1, be, D] soft feed
        # ================= HEAVY forward (SGLang) + DMax decode_uniform commit =================
        _t = _now(device)
        out = runner.forward(input_ids=None, position_ids=pos, inputs_embeds=inputs_embeds,
                             attention_mask=attn_bool, use_cache=False)
        h_sel, h_last = tap.pop()                                        # [1,be,m*D], [1,be,D]
        block_logits = out.logits.reshape(1, be, -1)[:, bs:be]           # [1, blk, V]
        stats.heavy_time += _now(device) - _t; stats.heavy_forwards += 1

        mask_idx = (x[0:1, bs:be] == MASK_ID)
        x0, high_conf_idx, _, breakflag = dmax_commit_uniform(block_logits, mask_idx, active, heavy_threshold)
        hci = high_conf_idx[0].nonzero(as_tuple=True)[0]
        if hci.numel() > 0:
            x[0, bs + hci] = x0[0, hci]; stats.heavy_commits += int(hci.numel())
        committed = active[0] & (x[0, bs:be] != MASK_ID)
        ci = committed.nonzero(as_tuple=True)[0]
        if ci.numel() > 0:
            block_embeds[0, ci] = _soft_embed(block_logits[0, ci], embed, MASK_ID, heavy_tau, heavy_top_k)

        if bool(breakflag) or not bool((x[0:1, bs:be] == MASK_ID).any()):
            break

        # ================= DRAFT forward (helper) on the tapped features =================
        block_x = x[0, bs:be]; mask_pos = (block_x == MASK_ID)
        committed_before = active[0] & (~mask_pos)
        draft_ids = x[:, bs:be].clone()
        if draft_committed_soft and bool(committed_before.any()):
            draft_ids[0][committed_before] = MASK_ID
        _t = _now(device)
        d = draft(input_ids=draft_ids, heavy_logits=block_logits, h_sel_denoise=h_sel[:, bs:be],
                  h_last_denoise=h_last[:, bs:be], h_sel_prefix=(h_sel[:, :bs] if bs > 0 else None),
                  attention_mask=None, position_ids=None, denoise_mask=None, tau=tau)
        stats.draft_time += _now(device) - _t; stats.draft_forwards += 1
        dlogits, dconf = d["logits"], d["conf"]
        if dconf is None:
            it += 1; continue
        darg = dlogits[0].argmax(-1); dc = dconf[0]
        # EXTEND: left-to-right prefix commit of masked slots while conf >= draft_threshold (>=1 for progress)
        mloc = mask_pos.nonzero(as_tuple=True)[0]
        if mloc.numel() > 0:
            ok = dc[mloc] >= draft_threshold
            keep = ~(torch.cumsum((~ok).long(), 0) > 0); keep[0] = True
            sel = mloc[keep]
            x[0, bs + sel] = darg[sel]
            block_embeds[0, sel] = _soft_embed(dlogits[0][sel], embed, MASK_ID, draft_tau, draft_top_k)
            stats.draft_commits += int(sel.numel())
        # FIX: override a committed slot only if conf-head >= draft_threshold AND the draft disagrees
        if draft_fix and bool(committed_before.any()):
            fix = committed_before & (dc >= draft_threshold) & (darg != block_x)
            floc = fix.nonzero(as_tuple=True)[0]
            if floc.numel() > 0:
                x[0, bs + floc] = darg[floc]
                block_embeds[0, floc] = _soft_embed(dlogits[0][floc], embed, MASK_ID, draft_tau, draft_top_k)
                stats.draft_fixes += int(floc.numel())
        it += 1

    # safety: never leave a [MASK] in the output
    still = (x[0:1, bs:be] == MASK_ID)
    if still.any() and block_logits is not None:
        sp = still[0].nonzero(as_tuple=True)[0]
        x[0, bs + sp] = block_logits[0, sp].argmax(dim=-1)


@torch.no_grad()
def generate_dbet_sglang(runner, tap, draft, prompt_ids, gen_length, block_length,
                         heavy_threshold=0.9, draft_threshold=0.9, max_iter_per_block=32, tau=None,
                         early_stop=True, heavy_tau=1.0, heavy_top_k=1, draft_tau=1.0, draft_top_k=2,
                         draft_committed_soft=False, draft_fix=True):
    """Grid-aligned multi-block DBet generation on the SGLang heavy. Returns (response_ids [n], DbetGenerateStats),
    response excludes the prompt and is cut at the first EOS. `runner`=diffusion ModelRunner (cuda graphs OFF so the
    tap fires), `tap`=HeavyFeatureTap on the heavy layers + final norm, `draft`=standalone DbetDraftStack."""
    device = prompt_ids.device
    embed = draft.frozen_embed
    P = prompt_ids.shape[1]

    first_block_start = (P // block_length) * block_length
    end_target = P + gen_length
    num_blocks = (end_target - first_block_start + block_length - 1) // block_length
    L = first_block_start + num_blocks * block_length

    x = torch.full((1, L), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids

    stats = DbetGenerateStats()
    eos_cut = L
    _t_wall = _now(device)
    for b in range(num_blocks):
        bs = first_block_start + b * block_length
        be = bs + block_length
        attn_bool = _block_causal_bool(be, block_length, device)
        decode_block_dbet_sglang(runner, tap, draft, embed, x, bs, be, attn_bool,
                                 heavy_threshold, draft_threshold, max_iter_per_block, tau, stats,
                                 heavy_tau=heavy_tau, heavy_top_k=heavy_top_k,
                                 draft_tau=draft_tau, draft_top_k=draft_top_k,
                                 draft_committed_soft=draft_committed_soft, draft_fix=draft_fix)
        if early_stop:
            resp_lo = max(P, bs)
            seg = x[0, resp_lo:be]
            eos_pos = (seg == EOS_ID).nonzero(as_tuple=True)[0]
            if eos_pos.numel() > 0:
                eos_cut = resp_lo + int(eos_pos[0].item())
                if be < L:
                    x[0, be:] = PAD_ID
                break

    stats.wall_time = _now(device) - _t_wall
    return x[0, P:eos_cut].clone(), stats
