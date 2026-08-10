# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# LENS-SURFACE PROBE (pure DMax, NO model changes) -- trial 1 of arc_head/exp_lens_surface.md.
#
# Question: at every (denoise step d, layer l, block position p), how close is the identity logit-lens
# readout to (i) the finally-committed token t_c, (ii) the same-step final-layer view f, and
# (iii) the same layer's own view at convergence q_c? Plus: is mid-layer disagreement noise or stable
# "workspace" content? (stability + qualitative reading happen OFFLINE from the raw dump.)
#
# Design decisions (locked in chat, 2026-08-06):
#   * Single decode pass per sample at the real-inference threshold. t_c = the SAME decode's converged
#     tokens (self-referential; no golden pass, no gold labels anywhere).
#   * Hidden states of every step are buffered (block slice only) and ALL metrics are computed after the
#     block converges -- exact, no replay forwards.
#   * Reading forward (--reading_forward, default ON): after convergence, ONE extra forward with the block
#     inputs = clean t_c embeddings. Purpose: measure q_{l,conv} vs q_{l,R} per cell. If they agree, the
#     ARC teacher can be the free last forward and this flag is dropped forever; if they diverge at
#     late-committed positions (mask inputs in the last forward), the teacher must be the reading forward.
#   * RECORD EVERYTHING: per-cell raw records (scalars + top-10 ids/probs) written as npz shards; every
#     aggregation (standard frame, event-aligned frame, masked-vs-clean split, stability score, top-k
#     overlaps, heatmaps) is done offline. No online accumulators.
#   * Readout is strictly the identity logit lens: top layer uses out.logits VERBATIM; intermediate layers
#     use lm_head(final_norm(h)). The one-time norm-convention self-check is inherited (read-only, prints
#     which hidden_states convention the model uses; it cannot alter the decode).
#
# The decode primitives (dmax_commit_uniform, build_block_causal_mask, _soft_embed) are IMPORTED from the
# production code, so this probe decodes byte-for-byte like generate_dbet(use_draft=False) == pure DMax.
#
# Output: probe_lens_surface_shardNNN.npz files (struct-of-arrays; see RAW_FIELDS) + one _meta.json.
#
# Run (GPU box, merged DMax checkpoint):
#   python probe_lens_surface.py --model_path ../../LLaDA2.0-mini-moe-merge \
#       --out_dir runs/lens_surface --limit 150 --gen_length 256 --threshold 0.5
#   # quick smoke: --limit 4 --gen_length 128

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_T3_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DFACTORY = os.path.join(_T3_ROOT, "dFactory")
_DINFER_PY = os.path.abspath(os.path.join(_HERE, "..", "python"))
for _p in (_DINFER_PY, _DFACTORY, os.path.join(_DFACTORY, "VeOmni")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoTokenizer  # noqa: E402

from dinfer.decoding.generate_t3d import (  # noqa: E402
    build_block_causal_mask, dmax_commit_uniform, _soft_embed, MASK_ID, EOS_ID, PAD_ID)
from eval_tasks import load_task  # noqa: E402

TOPK = 10                 # raw top-k kept per cell (ids + probs)
MAX_ITERS = 32            # matches generate_dbet / probe_layer_readout
D_READING = 999           # sentinel step index for the reading forward in the raw dump
MASKED, CLEAN = 0, 1      # input state of a position ENTERING the probed forward

# struct-of-arrays raw record; one row per (sample, block, step, layer, position) cell
RAW_FIELDS = [
    ("sample", np.int32), ("block", np.int16), ("d", np.int16), ("layer", np.int16), ("pos", np.int16),
    ("state", np.int8),           # 0 masked / 1 clean input at this step
    ("commit_step", np.int16),    # step at which this position was committed (READING rows keep it too)
    ("delta", np.int16),          # d - commit_step (D_READING rows: 999)
    ("d_conv", np.int16),         # the block's last decode step index (same for all rows of a block)
    ("t_c", np.int32),            # converged token at this position
    ("in_tok", np.int32),         # token id fed at this position ENTERING this step (MASK_ID if masked;
                                  # t_c on the reading forward). Exact -- committed tokens can change.
    ("q_tc", np.float16),         # q(t_c) -- probability of the converged token under this readout
    ("rank_tc", np.int32),        # 0-indexed rank of t_c under this readout
    ("entropy", np.float16),      # entropy of q (nats)
    ("kl_f", np.float16),         # KL(f || q), f = same-step FINAL-layer distribution (0 for top layer)
    ("kl_conv", np.float16),      # KL(q_c || q), q_c = SAME layer at the last decode step (0 at d=d_conv)
    ("kl_read", np.float16),      # KL(q_R || q), q_R = SAME layer at the reading forward (-1 if disabled)
]
TOP_IDS_DT, TOP_PROBS_DT = np.int32, np.float16


def load_fused(model_path, device):
    """Vendored LLaDA2MoeModelLM with fused-MoE + sdpa (the DMax-eager path). model_path MUST be a MERGED
    checkpoint (moe_convertor.py -m merge). Verbatim from probe_layer_readout.load_fused."""
    from models.llada2_moe.configuration_llada2_moe import LLaDA2MoeConfig
    from models.llada2_moe import modeling_llada2_moe as _mod
    from models.llada2_moe.modeling_llada2_moe import LLaDA2MoeModelLM
    if _mod.fused_moe_forward is None:
        raise RuntimeError("fused-MoE kernel unavailable: ensure dFactory/VeOmni is importable.")
    hcfg = LLaDA2MoeConfig.from_pretrained(model_path, trust_remote_code=True)
    if not str(hcfg.model_type).endswith("_veomni"):
        hcfg.model_type = str(hcfg.model_type) + "_veomni"
    hcfg.moe_implementation = "fused"
    m = LLaDA2MoeModelLM.from_pretrained(
        model_path, config=hcfg, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa")
    return m.eval().to(device)


class ShardWriter:
    """Struct-of-arrays buffer -> compressed npz shards (one per --shard_samples samples)."""
    def __init__(self, out_dir, shard_samples):
        self.out_dir = out_dir
        self.shard_samples = shard_samples
        self.cols = {name: [] for name, _ in RAW_FIELDS}
        self.top_ids, self.top_probs = [], []
        self.contexts = []
        self.samples_in_shard, self.shard_idx, self.total_rows = 0, 0, 0
        os.makedirs(out_dir, exist_ok=True)

    def add_block(self, cols_np, top_ids_np, top_probs_np):
        for name, _ in RAW_FIELDS:
            self.cols[name].append(cols_np[name])
        self.top_ids.append(top_ids_np)
        self.top_probs.append(top_probs_np)

    def add_context(self, ctx):
        """Per-sample prompt/geometry, for the per-block dashboard (tiny; plain json)."""
        self.contexts.append(ctx)

    def end_sample(self):
        self.samples_in_shard += 1
        if self.samples_in_shard >= self.shard_samples:
            self.flush()

    def flush(self):
        if not self.top_ids:
            return
        arrs = {name: np.concatenate(self.cols[name]).astype(dt) for name, dt in RAW_FIELDS}
        arrs["top_ids"] = np.concatenate(self.top_ids).astype(TOP_IDS_DT)        # [rows, TOPK]
        arrs["top_probs"] = np.concatenate(self.top_probs).astype(TOP_PROBS_DT)  # [rows, TOPK]
        n = arrs["sample"].shape[0]
        path = os.path.join(self.out_dir, f"probe_lens_surface_shard{self.shard_idx:03d}.npz")
        np.savez_compressed(path, **arrs)
        if self.contexts:
            with open(os.path.join(self.out_dir,
                                   f"probe_lens_context_shard{self.shard_idx:03d}.json"), "w") as fh:
                json.dump(self.contexts, fh)
        self.total_rows += n
        print(f"[shard] wrote {path} rows={n} (total={self.total_rows})")
        self.shard_idx += 1
        self.samples_in_shard = 0
        self.cols = {name: [] for name, _ in RAW_FIELDS}
        self.top_ids, self.top_probs = [], []
        self.contexts = []


_NORM_CHECKED = False


def _check_norm_convention(hidden_last_blk, logits_blk, lm_head, final_norm):
    """One-time READ-ONLY self-check (inherited from probe_layer_readout): report whether
    hidden_states[-1] is pre- or post-final-norm. The top-layer readout uses out.logits verbatim
    regardless, so this cannot shift any result -- it only guards interpretation."""
    global _NORM_CHECKED
    if _NORM_CHECKED:
        return
    d_norm = (lm_head(final_norm(hidden_last_blk)).float() - logits_blk.float()).abs().max().item()
    d_raw = (lm_head(hidden_last_blk).float() - logits_blk.float()).abs().max().item()
    print(f"[probe] hidden_states[-1] convention: |lm_head(norm(h)) - logits|={d_norm:.4f}  "
          f"|lm_head(h) - logits|={d_raw:.4f}  -> last entry is "
          f"{'PRE-norm (needs norm)' if d_norm < d_raw else 'POST-norm (already normed)'}. "
          f"Top-layer readout uses out.logits directly regardless.")
    _NORM_CHECKED = True


@torch.no_grad()
def _readout(l, NL, hs_blk_l, logits_blk, lm_head, final_norm):
    """Identity logit lens for layer l as fp32 logits [blk, V]. Top layer: model's REAL logits verbatim."""
    if l == NL - 1:
        return logits_blk.float()
    return lm_head(final_norm(hs_blk_l.to(logits_blk.device))).float()


@torch.no_grad()
def process_block(sample_idx, block_idx, bs, be, P, valid_hi, x_final, commit_step, mask_states, in_toks,
                  hs_buf, logits_buf, hs_read, logits_read, NL, lm_head, final_norm, writer, device):
    """Compute all per-cell records for one converged block and hand them to the shard writer.
    hs_buf: list over steps of [NL, blk, D] (cpu bf16). logits_buf: list of [blk, V] (cpu bf16).
    hs_read/logits_read: same for the reading forward, or None. mask_states: [n_steps, blk] bool (cpu),
    True = position was MASK entering that step. valid_hi: first invalid abs position (after EOS)."""
    blk = be - bs
    n_steps = len(hs_buf)
    d_conv = n_steps - 1
    pos_abs = np.arange(bs, be)
    valid = (pos_abs >= P) & (pos_abs < valid_hi)
    vpos = np.nonzero(valid)[0]
    if vpos.size == 0:
        return
    t_c = x_final[bs:be].to(device)                                  # [blk] long
    t_c_np = t_c.cpu().numpy()
    commit_np = commit_step                                          # [blk] int (numpy)
    have_read = hs_read is not None
    steps_iter = list(range(n_steps)) + ([D_READING] if have_read else [])

    nv = vpos.size
    rows_per_layer = nv * len(steps_iter)
    total = rows_per_layer * NL
    cols = {name: np.empty(total, dtype=dt) for name, dt in RAW_FIELDS}
    top_ids = np.empty((total, TOPK), dtype=TOP_IDS_DT)
    top_probs = np.empty((total, TOPK), dtype=TOP_PROBS_DT)

    # hoist per-step logits to the device ONCE (otherwise they re-upload NL times each)
    logits_dev = [lb.to(device) for lb in logits_buf]                # n_steps x [blk, V] bf16
    logits_read_dev = logits_read.to(device) if have_read else None
    # per-step references f (final-layer log-probs), shared across layers -- precompute once
    logf_steps = [F.log_softmax(ld.float(), dim=-1) for ld in logits_dev]   # [blk, V] fp32 each
    logf_read = F.log_softmax(logits_read_dev.float(), dim=-1) if have_read else None

    r0 = 0
    for l in range(NL):
        # same-layer references: convergence step and (optional) reading forward
        q_conv_logits = _readout(l, NL, hs_buf[d_conv][l], logits_dev[d_conv], lm_head, final_norm)
        logq_conv = F.log_softmax(q_conv_logits, dim=-1)             # [blk, V]
        p_conv = logq_conv.exp()
        if have_read:
            q_read_logits = _readout(l, NL, hs_read[l], logits_read_dev, lm_head, final_norm)
            logq_read = F.log_softmax(q_read_logits, dim=-1)
            p_read = logq_read.exp()

        for d in steps_iter:
            if d == D_READING:
                logits_blk = logits_read_dev
                hs_l = hs_read[l]
                state_np = np.ones(blk, dtype=np.int8)               # clean t_c inputs by construction
                in_np = t_c_np
                logf = logf_read
            else:
                logits_blk = logits_dev[d]
                hs_l = hs_buf[d][l]
                state_np = (~mask_states[d]).astype(np.int8)         # 1 = clean/committed input
                in_np = in_toks[d]
                logf = logf_steps[d]

            q_logits = _readout(l, NL, hs_l, logits_blk, lm_head, final_norm)   # [blk, V] fp32
            logq = F.log_softmax(q_logits, dim=-1)
            q = logq.exp()

            gold_logit = q_logits.gather(1, t_c[:, None])            # [blk, 1]
            rank = (q_logits > gold_logit).sum(1)                    # [blk]
            q_tc = q.gather(1, t_c[:, None]).squeeze(1)
            ent = -(q * logq).sum(1)
            kl_f = (logf.exp() * (logf - logq)).sum(1)               # KL(f || q); ~0 at top layer
            kl_conv = (p_conv * (logq_conv - logq)).sum(1)           # KL(q_c || q); 0 at d = d_conv
            if have_read:
                kl_read = (p_read * (logq_read - logq)).sum(1)
            tp, ti = q.topk(TOPK, dim=-1)

            sl = slice(r0, r0 + nv)
            cols["sample"][sl] = sample_idx
            cols["block"][sl] = block_idx
            cols["d"][sl] = d
            cols["layer"][sl] = l
            cols["pos"][sl] = vpos
            cols["state"][sl] = state_np[vpos]
            cols["commit_step"][sl] = commit_np[vpos]
            if d == D_READING:
                cols["delta"][sl] = np.full(nv, D_READING, dtype=np.int16)
            else:
                cols["delta"][sl] = (d - commit_np[vpos].astype(np.int32)).astype(np.int16)
            cols["d_conv"][sl] = d_conv
            cols["t_c"][sl] = t_c_np[vpos]
            cols["in_tok"][sl] = in_np[vpos]
            cols["q_tc"][sl] = q_tc[vpos].cpu().numpy()
            cols["rank_tc"][sl] = rank[vpos].cpu().numpy()
            cols["entropy"][sl] = ent[vpos].cpu().numpy()
            cols["kl_f"][sl] = kl_f[vpos].cpu().numpy()
            cols["kl_conv"][sl] = kl_conv[vpos].cpu().numpy()
            cols["kl_read"][sl] = kl_read[vpos].cpu().numpy() if have_read else -1.0
            top_ids[sl] = ti[vpos].cpu().numpy()
            top_probs[sl] = tp[vpos].cpu().numpy()
            r0 += nv

    writer.add_block(cols, top_ids, top_probs)


@torch.no_grad()
def decode_and_probe(model, embed, lm_head, final_norm, prompt_ids, gen_length, block_length,
                     threshold, device, sample_idx, writer, reading_forward=True):
    """Faithful DMax heavy-only block decode (mirror of probe_layer_readout.decode_and_maybe_probe),
    buffering every step's block hidden states + logits, then computing all lens-surface records after
    each block converges. Returns (x_full, eos_cut, n_blocks_processed, steps_per_block list)."""
    NL_holder = {}
    P = prompt_ids.shape[1]
    first_block_start = (P // block_length) * block_length
    end_target = P + gen_length
    num_blocks = (end_target - first_block_start + block_length - 1) // block_length
    L = first_block_start + num_blocks * block_length
    x = torch.full((1, L), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids
    eos_cut = L
    steps_hist = []

    for b in range(num_blocks):
        bs = first_block_start + b * block_length
        be = bs + block_length
        blk = be - bs
        attn = build_block_causal_mask(be, block_length, dtype=embed.weight.dtype, device=device)
        active = (x[0:1, bs:be] == MASK_ID)
        prefix_embeds = embed(x[:, :bs])
        block_embeds = embed(x[:, bs:be]).clone()
        block_logits = None
        hs_buf, logits_buf, mask_states, in_toks = [], [], [], []
        commit_step = np.full(blk, -1, dtype=np.int16)

        it = 0
        while it < MAX_ITERS:
            inputs_embeds = torch.cat([prefix_embeds, block_embeds], dim=1)
            mask_before = (x[0, bs:be] == MASK_ID).clone()
            in_toks.append(x[0, bs:be].detach().cpu().numpy().copy())   # exact input ids for this forward
            out = model(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=False,
                        output_hidden_states=True, return_dict=True)
            block_logits = out.logits[:, bs:be]
            NL_holder["NL"] = len(out.hidden_states)
            _check_norm_convention(out.hidden_states[-1][0, bs:be], block_logits[0], lm_head, final_norm)
            hs_buf.append(torch.stack([h[0, bs:be] for h in out.hidden_states]).to("cpu"))   # [NL, blk, D]
            logits_buf.append(block_logits[0].to("cpu"))                                     # [blk, V]
            mask_states.append(mask_before.cpu().numpy())

            # ---- DMax decode_uniform commit + committed re-decode + breakflag (byte-identical) ----
            curr = x[0, bs:be].clone()
            mask_idx = (curr == MASK_ID).unsqueeze(0)
            x0, high_conf_idx, max_probs, _ = dmax_commit_uniform(block_logits, mask_idx, active, threshold)
            x0b, hci = x0[0], high_conf_idx[0]
            ui = (hci | (active[0] & (curr != MASK_ID))).nonzero(as_tuple=True)[0]
            changed_any = bool((x0b[ui] != curr[ui]).any()) if ui.numel() > 0 else False
            if ui.numel() > 0:
                x[0, bs + ui] = x0b[ui]
            newly = mask_before.cpu().numpy() & (x[0, bs:be] != MASK_ID).cpu().numpy()
            commit_step[newly & (commit_step < 0)] = it
            breakflag = bool((max_probs[0][active[0]] >= 0.9).all()) or (not changed_any)
            block_embeds = embed(x[:, bs:be]).clone()
            soft = (active[0] & (x[0, bs:be] != MASK_ID)).nonzero(as_tuple=True)[0]
            if soft.numel() > 0:
                block_embeds[0, soft] = _soft_embed(block_logits[0, soft], embed, MASK_ID, 1.0, 1)
            if breakflag:
                break
            it += 1

        still = (x[0:1, bs:be] == MASK_ID)                            # safety: never leave a MASK
        if still.any() and block_logits is not None:
            sp = still[0].nonzero(as_tuple=True)[0]
            x[0, bs + sp] = block_logits[0, sp].argmax(dim=-1)
        commit_step[commit_step < 0] = len(hs_buf) - 1                # safety-filled / prompt-tail positions

        # EOS handling (== generate_dbet): find cut BEFORE deciding the valid range for this block
        resp_lo = max(P, bs)
        seg = x[0, resp_lo:be]
        eos_pos = (seg == EOS_ID).nonzero(as_tuple=True)[0]
        block_valid_hi = be
        stop = False
        if eos_pos.numel() > 0:
            eos_cut = resp_lo + int(eos_pos[0].item())
            block_valid_hi = eos_cut + 1                              # include the EOS token itself
            if be < L:
                x[0, be:] = PAD_ID
            stop = True

        # optional reading forward: clean t_c embeddings for the whole block, same mask
        hs_read = logits_read = None
        if reading_forward:
            read_embeds = torch.cat([embed(x[:, :bs]), embed(x[:, bs:be])], dim=1)
            out_r = model(inputs_embeds=read_embeds, attention_mask=attn, use_cache=False,
                          output_hidden_states=True, return_dict=True)
            hs_read = torch.stack([h[0, bs:be] for h in out_r.hidden_states]).to("cpu")
            logits_read = out_r.logits[0, bs:be].to("cpu")

        process_block(sample_idx, b, bs, be, P, block_valid_hi, x[0].cpu(), commit_step,
                      np.stack(mask_states), np.stack(in_toks), hs_buf, logits_buf, hs_read, logits_read,
                      NL_holder["NL"], lm_head, final_norm, writer, device)
        steps_hist.append(len(hs_buf))
        del hs_buf, logits_buf, hs_read, logits_read
        if stop:
            break
    return x[0].clone(), eos_cut, len(steps_hist), steps_hist


def main():
    p = argparse.ArgumentParser(description="Lens-surface probe, trial 1 (identity lens, record everything)")
    p.add_argument("--model_path", required=True, help="MERGED DMax checkpoint (moe_convertor.py -m merge)")
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--out_dir", required=True, help="directory for npz shards + meta json")
    p.add_argument("--task", choices=["gsm8k", "math500", "algebra", "asdiv"], default="gsm8k")
    p.add_argument("--gen_length", type=int, default=256)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.5, help="real-inference commit gate (single pass)")
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--shard_samples", type=int, default=25)
    p.add_argument("--no_reading_forward", action="store_true",
                   help="skip the clean-t_c reading forward (q_R reference)")
    p.add_argument("--gt_jsonl_path", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    mp = os.path.abspath(args.model_path)
    tok = AutoTokenizer.from_pretrained(os.path.abspath(args.tokenizer_path or mp), trust_remote_code=True)
    model = load_fused(mp, device)
    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    final_norm = model.model.norm
    V = int(model.config.vocab_size)
    print(f"[probe] layers={model.config.num_hidden_layers} vocab={V} block={args.block_length} "
          f"gen={args.gen_length} thr={args.threshold} reading_forward={not args.no_reading_forward}")

    rows = load_task(args.task, limit=args.limit, gt_jsonl_path=args.gt_jsonl_path)
    writer = ShardWriter(args.out_dir, args.shard_samples)
    t0 = time.time()
    all_steps = []
    for i, row in enumerate(rows):
        msgs = [{"role": "user", "content": row["prompt"]}]
        pid = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt").to(device)
        x_full, eos_cut, nb, steps_hist = decode_and_probe(
            model, embed, lm_head, final_norm, pid, args.gen_length, args.block_length,
            args.threshold, device, i, writer, reading_forward=not args.no_reading_forward)
        P = int(pid.shape[1])
        writer.add_context(dict(
            sample=i, P=P, block_length=args.block_length, eos_cut=int(eos_cut),
            first_block_start=(P // args.block_length) * args.block_length,
            prompt_ids=pid[0].tolist(),
            gen_ids=x_full[P:min(eos_cut + 1, x_full.shape[0])].tolist(),
        ))
        writer.end_sample()
        all_steps += steps_hist
        if i < 3 or (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(rows)}] blocks={nb} steps/block={steps_hist} eos_cut={eos_cut} "
                  f"rows={writer.total_rows} elapsed={time.time()-t0:.0f}s")
    writer.flush()

    meta = dict(model_path=mp, task=args.task, vocab=V, block_length=args.block_length,
                gen_length=args.gen_length, threshold=args.threshold, n_examples=len(rows),
                topk=TOPK, d_reading=D_READING, reading_forward=not args.no_reading_forward,
                fields=[f for f, _ in RAW_FIELDS] + ["top_ids", "top_probs"],
                steps_per_block_mean=float(np.mean(all_steps)) if all_steps else None,
                steps_per_block_hist=np.bincount(all_steps).tolist() if all_steps else None,
                total_rows=writer.total_rows)
    with open(os.path.join(args.out_dir, "probe_lens_surface_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[probe] done {time.time()-t0:.0f}s -> {args.out_dir} rows={writer.total_rows} "
          f"mean_steps/block={meta['steps_per_block_mean']}")


if __name__ == "__main__":
    main()
