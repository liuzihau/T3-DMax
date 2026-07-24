# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# LAYER-READOUT PROBE (pure DMax, NO drafter) -- prune-validity go/no-go for the intermediate Markov head.
#
# Question: at each transformer layer, does a cheap logit-lens readout already contain the token the model
# will finally commit, inside a SMALL candidate set? If some mid-layer does, an intermediate LM-head +
# top-p/top-500 prune + Markov-bias head is worth building; if only the last layer does, it is not.
#
# Method (matches the design agreed in chat):
#   * Golden pass  -- run the faithful DMax heavy-only block decode at a CONSERVATIVE threshold (default 0.7)
#                     to convergence. Its committed answer is the ONLY gold target (the model's own output).
#   * Experiment pass -- run the SAME decode at the real-inference threshold (default 0.5). For the FIRST 3
#                     forwards of each block, apply the FINAL lm_head (after the final RMSNorm = standard
#                     logit lens; NO training) to EVERY layer's hidden state at every block position, and
#                     score the gold token: rank, recall@K, percentile-recall, and membership in the
#                     nucleus(0.5) intersect top-500 set. Each position is tagged revealed / masked.
#   * Split every statistic by (layer, forward in {1,2,3}, position 0..blk-1, revealed/masked).
#   * Conditional (Markov) path: at position i, restrict to cases where positions 0..i-1 were all gold-top-1,
#     then recall@{1,5,10,50,100} at i -- does error compound left-to-right?
#   * Agreement diagnostic: fraction of positions where the 0.5-decode's final token == the 0.7 gold
#     (guards the two-threshold design; if low, recall is understated).
#
# The decode primitives (dmax_commit_uniform, build_block_causal_mask, _soft_embed) are IMPORTED from the
# production code, so this probe decodes byte-for-byte like generate_dbet(use_draft=False) == pure DMax.
#
# Output: an .npz of accumulator arrays + a meta .json. Plot with probe_layer_plot.py.
#
# Run (on the GPU box with the merged DMax checkpoint):
#   python probe_layer_readout.py --model_path ../../LLaDA2.0-mini-moe-merge \
#       --out_path runs/probe_readout.npz --limit 150 --gen_length 256 \
#       --exp_threshold 0.5 --gold_threshold 0.7
#   # quick smoke: --limit 8

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_T3_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_DFACTORY = os.path.join(_T3_ROOT, "dFactory")
_DINFER_PY = os.path.abspath(os.path.join(_HERE, "..", "python"))
# VeOmni before models.llada2_moe (its module-level fused_moe import is try/excepted to None otherwise)
for _p in (_DINFER_PY, _DFACTORY, os.path.join(_DFACTORY, "VeOmni")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoTokenizer  # noqa: E402

from dinfer.decoding.generate_t3d import build_block_causal_mask, dmax_commit_uniform  # noqa: E402
from dinfer.decoding.generate_dbet import _soft_embed, MASK_ID, EOS_ID, PAD_ID  # noqa: E402
from eval_tasks import load_task  # noqa: E402

# recall cut-offs (rank is 0-indexed; recall@K hit == gold_rank < K)
K_ABS = [1, 5, 10, 50, 100, 200, 500]
K_PCT = [0.0001, 0.001, 0.01, 0.05, 0.10, 0.25, 0.50]   # fractions of vocab
K_COND = [1, 5, 10, 50, 100]
N_FWD = 3                                                # probe the first 3 forwards of each block
NUC_P = 0.5                                              # nucleus mass
NUC_CAP = 500                                            # nucleus cap (candidate-set ceiling)


def load_fused(model_path, device):
    """Vendored LLaDA2MoeModelLM with fused-MoE + sdpa (the DMax-eager path). model_path MUST be a MERGED
    checkpoint (moe_convertor.py -m merge). Copied from eval_llada_gsm8k.load_fused -- no drafter."""
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


class Acc:
    """Lazy accumulators, allocated once we know the layer count. All indexed [layer, forward, pos, state]
    where state 0 = masked, 1 = revealed. Conditional arrays are [.., layer, forward, pos] (no state split)."""
    def __init__(self):
        self.ready = False

    def alloc(self, NL, F, P):
        z = lambda *s: np.zeros(s, dtype=np.float64)
        self.NL, self.F, self.P = NL, F, P
        self.count = z(NL, F, P, 2)
        self.hit_abs = z(len(K_ABS), NL, F, P, 2)
        self.hit_pct = z(len(K_PCT), NL, F, P, 2)
        self.hit_nuc = z(NL, F, P, 2)      # gold in nucleus(0.5) cap-500 set
        self.size_sum = z(NL, F, P, 2)     # sum of |S|
        self.sat_sum = z(NL, F, P, 2)      # count where nucleus saturated the cap (mass<0.5 within cap)
        self.rank_sum = z(NL, F, P, 2)     # sum of gold rank (for mean-rank sanity)
        self.cond_count = z(NL, F, P)      # cases where positions 0..i-1 were all gold-top-1
        self.cond_hit = z(len(K_COND), NL, F, P)
        self.agree = np.zeros(2, dtype=np.float64)   # [agree_count, total]  (0.5-final == 0.7-gold)
        self.ready = True

    def save(self, path, meta):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.savez_compressed(
            path, count=self.count, hit_abs=self.hit_abs, hit_pct=self.hit_pct, hit_nuc=self.hit_nuc,
            size_sum=self.size_sum, sat_sum=self.sat_sum, rank_sum=self.rank_sum,
            cond_count=self.cond_count, cond_hit=self.cond_hit, agree=self.agree)
        with open(os.path.splitext(path)[0] + "_meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)


@torch.no_grad()
def decode_and_maybe_probe(model, embed, lm_head, final_norm, prompt_ids, gen_length, block_length,
                           threshold, V, device, gold_block_fn=None, acc=None):
    """Faithful DMax heavy-only block decode (mirror of generate_dbet(use_draft=False) -> decode_block_heavy).
    If gold_block_fn is given (experiment pass), probe the first N_FWD forwards of each block against gold.
    gold_block_fn(bs, be) -> (gold_tokens[blk] long | None per pos, valid_mask[blk] bool). Returns
    (x_full[L] long, eos_cut int)."""
    P = prompt_ids.shape[1]
    first_block_start = (P // block_length) * block_length
    end_target = P + gen_length
    num_blocks = (end_target - first_block_start + block_length - 1) // block_length
    L = first_block_start + num_blocks * block_length
    x = torch.full((1, L), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids
    eos_cut = L
    probing = gold_block_fn is not None

    for b in range(num_blocks):
        bs = first_block_start + b * block_length
        be = bs + block_length
        attn = build_block_causal_mask(be, block_length, dtype=embed.weight.dtype, device=device)
        active = (x[0:1, bs:be] == MASK_ID)                       # decode region (excludes any prompt tail)
        prefix_embeds = embed(x[:, :bs])
        block_embeds = embed(x[:, bs:be]).clone()
        block_logits = None
        gold_blk = valid_blk = None
        if probing:
            gold_blk, valid_blk = gold_block_fn(bs, be)           # tensors on device or None

        it = 0
        while it < 32:                                            # max_iters (matches generate_dbet default)
            inputs_embeds = torch.cat([prefix_embeds, block_embeds], dim=1)
            want_hs = probing and it < N_FWD and gold_blk is not None and bool(valid_blk.any())
            mask_before = (x[0, bs:be] == MASK_ID).clone()        # position state ENTERING this forward
            out = model(inputs_embeds=inputs_embeds, attention_mask=attn, use_cache=False,
                        output_hidden_states=want_hs, return_dict=True)
            block_logits = out.logits[:, bs:be]

            if want_hs:
                _probe_forward(out.hidden_states, block_logits, bs, be, lm_head, final_norm, V,
                               gold_blk, valid_blk, mask_before, it, acc)

            # ---- DMax decode_uniform commit + committed re-decode + breakflag (== decode_block_heavy) ----
            curr = x[0, bs:be].clone()
            mask_idx = (curr == MASK_ID).unsqueeze(0)
            x0, high_conf_idx, max_probs, _ = dmax_commit_uniform(block_logits, mask_idx, active, threshold)
            x0b, hci = x0[0], high_conf_idx[0]
            ui = (hci | (active[0] & (curr != MASK_ID))).nonzero(as_tuple=True)[0]
            changed_any = bool((x0b[ui] != curr[ui]).any()) if ui.numel() > 0 else False
            if ui.numel() > 0:
                x[0, bs + ui] = x0b[ui]
            breakflag = bool((max_probs[0][active[0]] >= 0.9).all()) or (not changed_any)
            block_embeds = embed(x[:, bs:be]).clone()
            soft = (active[0] & (x[0, bs:be] != MASK_ID)).nonzero(as_tuple=True)[0]
            if soft.numel() > 0:
                block_embeds[0, soft] = _soft_embed(block_logits[0, soft], embed, MASK_ID, 1.0, 1)
            if breakflag:
                break
            it += 1

        still = (x[0:1, bs:be] == MASK_ID)                        # safety: never leave a MASK
        if still.any() and block_logits is not None:
            sp = still[0].nonzero(as_tuple=True)[0]
            x[0, bs + sp] = block_logits[0, sp].argmax(dim=-1)

        resp_lo = max(P, bs)                                      # early stop at first EOS (== generate_dbet)
        seg = x[0, resp_lo:be]
        eos_pos = (seg == EOS_ID).nonzero(as_tuple=True)[0]
        if eos_pos.numel() > 0:
            eos_cut = resp_lo + int(eos_pos[0].item())
            if be < L:
                x[0, be:] = PAD_ID
            break
    return x[0].clone(), eos_cut


_NORM_CHECKED = False   # one-time convention self-check flag


@torch.no_grad()
def _probe_forward(hidden_states, logits_final, bs, be, lm_head, final_norm, V,
                   gold_blk, valid_blk, mask_before, f, acc):
    """Score every layer's logit-lens readout for one forward against the gold block. Accumulates into acc.
    logits_final = the model's REAL block logits (out.logits[:,bs:be]) -- used verbatim for the top layer so
    the last-layer readout is guaranteed identical to the model (no dependence on the hidden_states norm
    convention). Intermediate layers use the standard logit lens: lm_head(final_norm(h))."""
    global _NORM_CHECKED
    NL = len(hidden_states)
    if not acc.ready:
        acc.alloc(NL, N_FWD, be - bs)
    gold = gold_blk                                              # [blk] long (device)
    valid = valid_blk                                           # [blk] bool (device)
    state = (~mask_before).long()                               # 1 = revealed, 0 = masked  [blk]
    vpos = valid.nonzero(as_tuple=True)[0]
    if vpos.numel() == 0:
        return
    vpos_np = vpos.cpu().numpy()
    state_np = state[vpos].cpu().numpy()
    gold_v = gold[vpos]
    correct1_full = torch.zeros(be - bs, dtype=torch.bool, device=gold.device)   # top-1 correct, ALL positions

    if not _NORM_CHECKED:                                        # print which hidden_states convention is in use
        hl = hidden_states[-1][0, bs:be, :]
        d_norm = (lm_head(final_norm(hl)).float() - logits_final[0]).abs().max().item()
        d_raw = (lm_head(hl).float() - logits_final[0]).abs().max().item()
        print(f"[probe] hidden_states[-1] convention: |lm_head(norm(h)) - logits|={d_norm:.4f}  "
              f"|lm_head(h) - logits|={d_raw:.4f}  -> last entry is "
              f"{'PRE-norm (needs norm)' if d_norm < d_raw else 'POST-norm (already normed)'}. "
              f"Top-layer readout uses out.logits directly regardless.")
        _NORM_CHECKED = True

    for L in range(NL):
        if L == NL - 1:
            logits = logits_final[0].float()                   # model's REAL logits -- unambiguous top layer
        else:
            h = hidden_states[L][0, bs:be, :]                  # [blk, D] intermediate: standard logit lens
            logits = lm_head(final_norm(h)).float()            # [blk, V]
        lv = logits[vpos]                                       # [nv, V] valid positions only
        gl = lv.gather(1, gold_v[:, None])                      # [nv,1] gold logit
        rank = (lv > gl).sum(1)                                 # [nv] 0-indexed rank of gold
        top1 = lv.argmax(1)
        correct1_full[vpos] = (top1 == gold_v)                  # for the conditional path (this layer/forward)
        # nucleus(0.5) cap-500 set
        topp = torch.softmax(lv, dim=1).topk(min(NUC_CAP, V), dim=1).values   # [nv, cap] desc
        cum = topp.cumsum(1)
        reach = cum >= NUC_P
        never = ~reach.any(1)
        size = torch.where(never, torch.full_like(reach[:, 0].long(), NUC_CAP),
                           reach.float().argmax(1).long() + 1).clamp(max=NUC_CAP)   # |S|
        gold_in = rank < size                                   # gold within the nucleus set

        rank_np = rank.cpu().numpy()
        size_np = size.cpu().numpy().astype(np.float64)
        sat_np = never.cpu().numpy().astype(np.float64)
        gin_np = gold_in.cpu().numpy().astype(np.float64)
        idx = (L, f, vpos_np, state_np)
        np.add.at(acc.count, idx, 1.0)
        np.add.at(acc.rank_sum, idx, rank_np.astype(np.float64))
        np.add.at(acc.size_sum, idx, size_np)
        np.add.at(acc.sat_sum, idx, sat_np)
        np.add.at(acc.hit_nuc, idx, gin_np)
        for ki, K in enumerate(K_ABS):
            np.add.at(acc.hit_abs[ki], idx, (rank_np < K).astype(np.float64))
        for ki, q in enumerate(K_PCT):
            kr = max(1, int(q * V))
            np.add.at(acc.hit_pct[ki], idx, (rank_np < kr).astype(np.float64))

        # conditional (Markov) path: at valid position i, require all valid j<i to be gold-top-1
        c = correct1_full.cpu().numpy()
        vmask = valid.cpu().numpy()
        rank_by_pos = {int(p): int(r) for p, r in zip(vpos_np, rank_np)}
        for i in range(be - bs):
            if not vmask[i]:
                continue
            prefix = [c[j] for j in range(i) if vmask[j]]
            if not all(prefix):                                 # empty prefix -> all() True (i=0 case)
                continue
            acc.cond_count[L, f, i] += 1.0
            ri = rank_by_pos[i]
            for ki, K in enumerate(K_COND):
                if ri < K:
                    acc.cond_hit[ki, L, f, i] += 1.0


def main():
    p = argparse.ArgumentParser(description="Layer-readout prune-validity probe (pure DMax, no drafter)")
    p.add_argument("--model_path", required=True, help="MERGED DMax checkpoint (moe_convertor.py -m merge)")
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--out_path", required=True, help=".npz output (a _meta.json is written alongside)")
    p.add_argument("--task", choices=["gsm8k", "math500", "algebra", "asdiv"], default="gsm8k")
    p.add_argument("--gen_length", type=int, default=256)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--exp_threshold", type=float, default=0.5, help="real-inference commit gate (experiment)")
    p.add_argument("--gold_threshold", type=float, default=0.7, help="conservative gate for the gold target")
    p.add_argument("--limit", type=int, default=150)
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
    NL = int(model.config.num_hidden_layers)
    print(f"[probe] model layers={NL} vocab={V} block={args.block_length} gen={args.gen_length} "
          f"exp_thr={args.exp_threshold} gold_thr={args.gold_threshold}")

    rows = load_task(args.task, limit=args.limit, gt_jsonl_path=args.gt_jsonl_path)
    acc = Acc()
    t0 = time.time()
    for i, row in enumerate(rows):
        msgs = [{"role": "user", "content": row["prompt"]}]
        pid = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_tensors="pt").to(device)
        # (1) golden pass: conservative decode -> gold tokens (the model's own converged answer)
        gold_x, gold_eos = decode_and_maybe_probe(
            model, embed, lm_head, final_norm, pid, args.gen_length, args.block_length,
            args.gold_threshold, V, device, gold_block_fn=None, acc=None)

        def gold_block_fn(bs, be, _gx=gold_x, _ge=gold_eos, _P=pid.shape[1]):
            pos = torch.arange(bs, be, device=device)
            valid = (pos >= _P) & (pos < _ge)                  # generated region, before gold EOS
            g = _gx[bs:be]
            return g, valid

        # (2) experiment pass: real-inference decode, probe first 3 forwards/block against gold
        exp_x, exp_eos = decode_and_maybe_probe(
            model, embed, lm_head, final_norm, pid, args.gen_length, args.block_length,
            args.exp_threshold, V, device, gold_block_fn=gold_block_fn, acc=acc)

        # agreement diagnostic: 0.5-final vs 0.7-gold over the commonly-valid generated positions
        P = pid.shape[1]
        hi = min(gold_eos, exp_eos, gold_x.shape[0])
        if hi > P:
            g = gold_x[P:hi]
            e = exp_x[P:hi]
            acc.agree[0] += float((g == e).sum().item())
            acc.agree[1] += float(g.numel())

        if i < 3 or (i + 1) % 25 == 0:
            ag = acc.agree[0] / max(acc.agree[1], 1)
            print(f"[{i+1}/{len(rows)}] gold_eos={gold_eos} exp_eos={exp_eos} "
                  f"agree={ag:.3f} elapsed={time.time()-t0:.0f}s")

    meta = dict(model_path=mp, task=args.task, layers=NL, vocab=V, block_length=args.block_length,
                gen_length=args.gen_length, exp_threshold=args.exp_threshold,
                gold_threshold=args.gold_threshold, n_examples=len(rows),
                K_ABS=K_ABS, K_PCT=K_PCT, K_COND=K_COND, N_FWD=N_FWD, nucleus_p=NUC_P, nucleus_cap=NUC_CAP,
                agree=float(acc.agree[0] / max(acc.agree[1], 1)))
    acc.save(args.out_path, meta)
    # last-layer sanity: this readout == the model's own logits, so recall here is the model's true recall.
    if acc.ready:
        li = acc.NL - 1
        k1, k10 = K_ABS.index(1), K_ABS.index(10)
        def _r(ki, pslice):
            h = acc.hit_abs[ki, li, 0, pslice, MASKED].sum()
            c = acc.count[li, 0, pslice, MASKED].sum()
            return h / max(c, 1)
        print(f"[probe] LAST-LAYER F1 masked recall (== model's real recall): "
              f"pos0 @1={_r(k1, slice(0,1)):.3f} @10={_r(k10, slice(0,1)):.3f} | "
              f"all-pos @1={_r(k1, slice(None)):.3f} @10={_r(k10, slice(None)):.3f}  "
              f"(if pos0@1 is low with self-gold, the model genuinely can't call it at F1)")
    print(f"[probe] done {time.time()-t0:.0f}s -> {args.out_path} (+ _meta.json). "
          f"final agreement(exp vs gold)={meta['agree']:.3f}")
    print(f"[probe] plot: python {os.path.join(_HERE, 'probe_layer_plot.py')} --npz {args.out_path}")


if __name__ == "__main__":
    main()
