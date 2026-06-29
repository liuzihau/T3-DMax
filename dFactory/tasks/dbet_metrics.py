# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
"""DBet metrics + held-out validation (pure torch; no VeOmni internals so it is smoke-testable off-cluster).

Provides:
  * `MetricsLogger`     -- append-only JSONL (one record per logged step/eval) + optional wandb mirroring.
  * `roc_auc`           -- dependency-free ROC-AUC (Mann-Whitney U with average ranks for ties).
  * `load_holdout_examples` -- carve the last-N raw examples out of the train file -> (clean_ids, prompt_len).
  * `evaluate_dbet`     -- held-out eval with a sigma SWEEP, producing the four figures' data:
                             loss + drafter accuracy, confidence-head AUC, accuracy-vs-block-position,
                             accuracy-vs-mask-ratio.

The eval re-uses `dbet_forward` from dbet_train_core (the exact train-time forward), wrapped in no_grad, so the
validation numbers can never drift from what the model is actually trained on.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict, deque

import torch
import torch.nn.functional as F

from dbet_train_core import MASK_ID, dbet_forward, decay_weights


# ----------------------------------------------------------------------------- AUC (dependency-free)
def roc_auc(scores, labels) -> float:
    """ROC-AUC via the Mann-Whitney U statistic with average ranks for ties. Returns nan if a class is absent.
    scores: 1D float tensor/list of predicted positives; labels: 0/1."""
    # force CPU: AUC is cheap and ranking on CPU avoids device-mismatch when indexing (scores/labels are on cuda)
    s = torch.as_tensor(scores, dtype=torch.float64).flatten().cpu()
    y = torch.as_tensor(labels, dtype=torch.float64).flatten().cpu()
    n = y.numel()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0 or n == 0:
        return float("nan")
    order = torch.argsort(s)
    s_sorted = s[order]
    # average ranks (1-based), handling ties by assigning the mean rank within each tie group
    ranks = torch.empty(n, dtype=torch.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0                       # mean of 1-based positions i..j
        ranks[i:j + 1] = avg
        i = j + 1
    rank_of = torch.empty(n, dtype=torch.float64)
    rank_of[order] = ranks
    sum_ranks_pos = float(rank_of[y == 1].sum())
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# ----------------------------------------------------------------------------- logger
class MetricsLogger:
    """Append-only JSONL + optional wandb. `log(metrics, step, split, detail=)`: scalar `metrics` go to BOTH
    jsonl and wandb (prefixed `split/`); `detail` (nested lists, e.g. sweep arrays) goes to jsonl only."""

    def __init__(self, jsonl_path: str = "", use_wandb: bool = False, enabled: bool = True):
        self.enabled = enabled
        self.use_wandb = use_wandb
        self._f = None
        if enabled and jsonl_path:
            os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)
            self._f = open(jsonl_path, "a")
            self.path = jsonl_path

    def log(self, metrics: dict, step: int, split: str, detail: dict | None = None):
        if not self.enabled:
            return
        if self._f is not None:
            rec = {"step": int(step), "split": split, **metrics, **(detail or {})}
            self._f.write(json.dumps(rec) + "\n")
            self._f.flush()
        if self.use_wandb:
            import wandb
            scalars = {f"{split}/{k}": v for k, v in metrics.items()
                       if isinstance(v, (int, float)) and not (isinstance(v, float) and v != v)}
            if scalars:
                wandb.log(scalars, step=step)

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None


# ----------------------------------------------------------------------------- held-out data
def _read_last_n_messages(train_path: str, n: int, text_keys: str = "messages"):
    """Last-n raw examples' message lists. Supports a jsonl file (streamed via deque) or an HF dataset dir."""
    if os.path.isfile(train_path) and train_path.endswith((".jsonl", ".json")):
        buf = deque(maxlen=n)
        with open(train_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    buf.append(line)
        rows = []
        for line in buf:
            obj = json.loads(line)
            if text_keys in obj:
                rows.append(obj[text_keys])
        return rows
    # fall back to HF datasets (dir or hub id)
    from datasets import load_dataset
    ds = load_dataset(train_path, split="train")
    n = min(n, len(ds))
    sel = ds.select(range(len(ds) - n, len(ds)))
    return [ex[text_keys] for ex in sel]


def load_holdout_examples(train_path: str, n: int, tokenizer, max_seq_len: int, text_keys: str = "messages"):
    """Last-n raw examples -> list of (clean_ids [max_seq_len], prompt_length). NOTE: these are the *tail* of the
    train file; with shuffled streaming they may also appear in training (minor leakage) -- fine for a diagnostic
    curve. For a fully clean split, stage a separate val file and point this at it."""
    from dataset.data_transform_dbet import apply_chat_template_mdm
    rows = _read_last_n_messages(train_path, n, text_keys)
    out = []
    for messages in rows:
        ids, prompt_len = apply_chat_template_mdm(messages=messages, tokenizer=tokenizer, max_length=max_seq_len)
        out.append((ids, int(prompt_len)))
    return out


def _build_dual_stream(noisy_ids, clean_ids, mask_proto, device):
    """Mirror of the train loop's dual-stream assembly (train_dbet.py block-diffusion branch).
    noisy_ids/clean_ids: [B,L] -> micro_batch dict for dbet_forward."""
    B, L = noisy_ids.shape
    full = torch.cat([noisy_ids, clean_ids], dim=1).to(device)
    npos = torch.arange(L, device=device, dtype=torch.long)
    pos = torch.cat([npos, npos], dim=0).unsqueeze(0).expand(B, -1).clone()
    return {
        "input_ids": full,
        "noisy_input_ids": noisy_ids.to(device),
        "position_ids": pos,
        "attention_mask": mask_proto.expand(B, -1, -1, -1),
    }


# ----------------------------------------------------------------------------- evaluation (sigma sweep)
@torch.no_grad()
def evaluate_dbet(core, holdout, args, mask_proto, device, mask_id: int = MASK_ID,
                  sigmas=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """Held-out eval over `holdout` (list of (clean_ids, prompt_len)) re-noised at each sigma in `sigmas`.
    Returns (scalar_metrics, detail).

    Metrics on the remaining-masked positions (drafter argmax/top-k vs golden):
      acc                      pooled top-1 accuracy over ALL remaining positions (headline)
      top2 / top3              pooled top-2 / top-3 coverage (golden in the drafter's top-k) -- "if we suggest
                               3 tokens at a position, does golden appear?"
      acc6 / top2_6 / top3_6   the same but over the FIRST `eval_tpf` (default 6) remaining positions per block,
                               averaged ACROSS positions (per-position acc, then mean over k<tpf). The heavy
                               commits ~3-7 tokens/forward (~6 TPF), so this is the decision-relevant window.
      h2_acc / h2_acc6         HEAVY second-pass top-1 accuracy (re-forward the heavy on the committed sequence)
                               over all remaining / the first-tpf window -- the teacher ceiling the drafter
                               approximates (only when eval_heavy_second).
      auc / conf / tok         confidence-head ROC-AUC, conf BCE, decayed token CE.
    detail: acc_by_pos, acc_by_pos_topk, acc_by_sigma, auc_by_sigma, acc_by_pos_sigma."""
    from dataset.data_transform_dbet import block_left_to_right_reveal
    bs = args.train.block_size
    L = args.data.max_seq_len
    tpf = int(getattr(args.train, "eval_tpf", 6))                      # heavy tokens/forward window
    heavy_second = bool(getattr(args.train, "eval_heavy_second", True))

    all_conf, all_accept = [], []
    tok_num = tok_den = 0.0
    conf_num = conf_den = 0.0
    pos_total = defaultdict(float)
    pos_top1 = defaultdict(float); pos_top2 = defaultdict(float); pos_top3 = defaultdict(float)
    pos_h2 = defaultdict(float)
    ps_correct = defaultdict(float)        # joint (sigma, k) -> top-1 correct count  (for the acc heatmap)
    ps_total = defaultdict(float)
    pool = {"t1": 0.0, "t2": 0.0, "t3": 0.0, "h2": 0.0, "n": 0.0}
    sig_correct = {s: 0.0 for s in sigmas}
    sig_total = {s: 0.0 for s in sigmas}
    sig_conf = {s: [] for s in sigmas}
    sig_accept = {s: [] for s in sigmas}

    was_training = core.training
    core.eval()
    for clean_ids, prompt_len in holdout:
        clean_ids = clean_ids[:L]
        maskable = torch.arange(L) >= prompt_len
        for sigma in sigmas:
            noisy = block_left_to_right_reveal(clean_ids.clone(), (sigma, sigma), maskable, mask_id, bs)
            mb = _build_dual_stream(noisy.unsqueeze(0), clean_ids.unsqueeze(0), mask_proto, device)
            out = dbet_forward(core, mb, args, mask_id, return_post_commit=heavy_second)
            if heavy_second:
                logits, conf, remaining, golden, post_commit = out
            else:
                logits, conf, remaining, golden = out
            rem = remaining[0]
            if int(rem.sum()) == 0:
                continue
            g = golden[0]
            top3_idx = logits[0].topk(3, dim=-1).indices                  # [L,3]
            in1 = top3_idx[:, 0] == g
            in2 = (top3_idx[:, :2] == g.unsqueeze(-1)).any(-1)
            in3 = (top3_idx == g.unsqueeze(-1)).any(-1)
            correct = in1 & rem                                            # top-1 (== existing acc/heatmap)

            # heavy SECOND pass: re-forward the heavy with the committed tokens revealed -> teacher ceiling
            if heavy_second:
                mb2 = _build_dual_stream(post_commit, golden.unsqueeze(0) if golden.dim() == 1 else golden,
                                         mask_proto, device)
                h2 = core.heavy(input_ids=mb2["input_ids"], attention_mask=mb2["attention_mask"],
                                position_ids=mb2["position_ids"], use_cache=False, return_dict=True)
                h2_correct = (h2.logits[0, :L].argmax(-1) == g) & rem

            # decayed CE + BCE (train-equivalent aggregate, for the loss curve)
            w = decay_weights(remaining, bs)[0]
            denom = float(w.sum().clamp_min(1.0))
            ce = F.cross_entropy(logits[0], g, reduction="none")
            tok_num += float((ce * w).sum()); tok_den += denom

            # per-block distance index k (0 at first remaining pos in block)
            nb = L // bs
            k = (torch.cumsum(rem.view(nb, bs).long(), dim=-1) - 1).clamp(min=0).view(L)
            idxs = torch.nonzero(rem, as_tuple=False).flatten()
            kk = k[idxs].tolist()
            c1 = in1[idxs].tolist(); c2 = in2[idxs].tolist(); c3 = in3[idxs].tolist()
            hh = h2_correct[idxs].tolist() if heavy_second else [0.0] * len(kk)
            for j, p_k in enumerate(kk):
                pos_total[p_k] += 1.0
                pos_top1[p_k] += c1[j]; pos_top2[p_k] += c2[j]; pos_top3[p_k] += c3[j]; pos_h2[p_k] += hh[j]
                ps_total[(float(sigma), p_k)] += 1.0
                ps_correct[(float(sigma), p_k)] += c1[j]
            pool["t1"] += float(in1[rem].sum()); pool["t2"] += float(in2[rem].sum())
            pool["t3"] += float(in3[rem].sum()); pool["n"] += float(rem.sum())
            if heavy_second:
                pool["h2"] += float(h2_correct[rem].sum())

            sig_correct[sigma] += float(correct.sum())
            sig_total[sigma] += float(rem.sum())
            if conf is not None:
                cvals = conf[0][rem]
                avals = correct[rem].float()
                all_conf.append(cvals); all_accept.append(avals)
                sig_conf[sigma].append(cvals); sig_accept[sigma].append(avals)
                c = conf[0][rem].float().clamp(1e-5, 1 - 1e-5)
                a = correct[rem].float()
                bce = -(a * c.log() + (1 - a) * (1 - c).log())
                conf_num += float((bce * w[rem]).sum()); conf_den += denom
    if was_training:
        core.train()

    n = pool["n"] or 1.0

    def _avg_pos(table):  # avg-of-position over the first-tpf window (per-position rate, then mean over k<tpf)
        ks = [k for k in pos_total if k < tpf]
        return (sum(table[k] / pos_total[k] for k in ks) / len(ks)) if ks else float("nan")

    scalar = {
        "acc": pool["t1"] / n,                       # pooled top-1 over all remaining (headline)
        "top2": pool["t2"] / n, "top3": pool["t3"] / n,
        "acc6": _avg_pos(pos_top1),                  # first-tpf window, avg across positions
        "top2_6": _avg_pos(pos_top2), "top3_6": _avg_pos(pos_top3),
        "tok": tok_num / (tok_den or 1.0),
        "n": int(pool["n"]),
    }
    if heavy_second:
        scalar["h2_acc"] = pool["h2"] / n
        scalar["h2_acc6"] = _avg_pos(pos_h2)
    if all_conf:
        scalar["auc"] = roc_auc(torch.cat(all_conf), torch.cat(all_accept))
        scalar["conf"] = conf_num / (conf_den or 1.0)

    acc_by_sigma, auc_by_sigma = [], []
    for s in sigmas:
        tot = sig_total[s] or 1.0
        acc_s = sig_correct[s] / tot
        acc_by_sigma.append([float(s), acc_s, int(sig_total[s])])
        scalar[f"acc_sig{s}"] = acc_s
        if sig_conf[s]:
            auc_s = roc_auc(torch.cat(sig_conf[s]), torch.cat(sig_accept[s]))
            auc_by_sigma.append([float(s), auc_s])
            scalar[f"auc_sig{s}"] = auc_s

    acc_by_pos = [[k, pos_top1[k] / pos_total[k], int(pos_total[k])] for k in sorted(pos_total.keys())]
    # richer per-position table: [k, top1, top2, top3, h2, count]
    acc_by_pos_topk = [[k, pos_top1[k] / pos_total[k], pos_top2[k] / pos_total[k], pos_top3[k] / pos_total[k],
                        pos_h2[k] / pos_total[k], int(pos_total[k])] for k in sorted(pos_total.keys())]
    acc_by_pos_sigma = [[s, k, ps_correct[(s, k)] / ps_total[(s, k)], int(ps_total[(s, k)])]
                        for (s, k) in sorted(ps_total.keys())]
    detail = {"acc_by_pos": acc_by_pos, "acc_by_pos_topk": acc_by_pos_topk, "acc_by_sigma": acc_by_sigma,
              "auc_by_sigma": auc_by_sigma, "acc_by_pos_sigma": acc_by_pos_sigma}
    return scalar, detail
