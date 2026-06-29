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
    Returns (scalar_metrics, detail) where:
      scalar_metrics: acc, auc, tok (decayed CE, train-equivalent), conf (BCE), n + per-sigma acc/auc scalars
      detail: acc_by_pos [[k,acc,count]...], acc_by_sigma [[s,acc,count]...], auc_by_sigma [[s,auc]...]
    `acc` is UNWEIGHTED fraction of remaining-masked positions where drafter argmax == golden (headline)."""
    from dataset.data_transform_dbet import block_left_to_right_reveal
    bs = args.train.block_size
    L = args.data.max_seq_len

    all_conf, all_accept = [], []
    tok_num = tok_den = 0.0
    conf_num = conf_den = 0.0
    pos_correct = defaultdict(float)
    pos_total = defaultdict(float)
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
            logits, conf, remaining, golden = dbet_forward(core, mb, args, mask_id)
            rem = remaining[0]
            if int(rem.sum()) == 0:
                continue
            pred = logits[0].argmax(-1)
            correct = ((pred == golden[0]) & rem)

            # decayed CE + BCE (train-equivalent aggregate, for the loss curve)
            w = decay_weights(remaining, bs)[0]
            denom = float(w.sum().clamp_min(1.0))
            ce = F.cross_entropy(logits[0], golden[0], reduction="none")
            tok_num += float((ce * w).sum()); tok_den += denom

            # per-block distance index k (0 at first remaining pos in block)
            nb = L // bs
            rem_b = rem.view(nb, bs)
            k = (torch.cumsum(rem_b.long(), dim=-1) - 1).clamp(min=0).view(L)
            corr_b = correct.float()
            idxs = torch.nonzero(rem, as_tuple=False).flatten()
            for p in idxs.tolist():
                kk = int(k[p])
                pos_total[kk] += 1.0
                pos_correct[kk] += float(corr_b[p])

            sig_correct[sigma] += float(correct.sum())
            sig_total[sigma] += float(rem.sum())
            if conf is not None:
                cvals = conf[0][rem]
                avals = correct[rem].float()
                all_conf.append(cvals); all_accept.append(avals)
                sig_conf[sigma].append(cvals); sig_accept[sigma].append(avals)
                c = conf[0][rem].clamp(1e-5, 1 - 1e-5)
                a = correct[rem].float()
                bce = -(a * c.log() + (1 - a) * (1 - c).log())
                conf_num += float((bce * w[rem]).sum()); conf_den += denom
    if was_training:
        core.train()

    tot_correct = sum(sig_correct.values())
    tot_total = sum(sig_total.values()) or 1.0
    scalar = {
        "acc": tot_correct / tot_total,
        "tok": tok_num / (tok_den or 1.0),
        "n": int(tot_total),
    }
    if all_conf:
        conf_cat = torch.cat(all_conf); accept_cat = torch.cat(all_accept)
        scalar["auc"] = roc_auc(conf_cat, accept_cat)
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

    acc_by_pos = [[k, pos_correct[k] / pos_total[k], int(pos_total[k])]
                  for k in sorted(pos_total.keys())]
    detail = {"acc_by_pos": acc_by_pos, "acc_by_sigma": acc_by_sigma, "auc_by_sigma": auc_by_sigma}
    return scalar, detail
