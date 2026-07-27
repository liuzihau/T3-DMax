# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# DARC training-data transform. Turns ONE collected self-gold record (prompt_ids + gold_ids, produced by
# dFactory/scripts/collect_gold_data.py) into the training tensors the DARC forward consumes. Borrows the
# DMax/DBet output schema and masking conventions verbatim (data_transform.process_mdm_tokenized_example):
#
#   {input_ids, noisy_input_ids, attention_mask, labels}    (all [max_seq_len], long; +prompt_len scalar)
#
# Differences from the DMax tokenized transform (kept minimal on purpose):
#   * Input is ALREADY tokenized (prompt_ids + gold_ids) -> no chat template.
#   * FIRST-TRIAL noising = the WHOLE generated region masked (sigma=1, single forward). Other modes
#     ('ltr_reveal', 'random') reuse the DBet/DMax noisers for later multi-forward training.
#   * Loss follows the LLaDA masked-token objective (loss only where noisy==MASK), identical to DMax:
#         loss_positions = (noisy_input_ids == MASK_ID);  labels[~loss_positions] = -100.
#     For 'all_masked' this means loss on every generated position; padding & prompt are never masked -> -100.
#
# The per-block SEED (block-relative position 0 = the frozen `s_0`) is NOT excluded here -- that belongs to
# the DARC model forward (it must match inference), which skips Loss-1 at positions where (pos % block_size)==0.
# Revealed positions in a partly-prompt first block naturally carry HARD (real) tokens in noisy_input_ids,
# which is exactly the "hard embed for revealed positions" the design wants.
#
# Self-test (no model, no `datasets` needed):  python data_transform_darc.py

import json
import os
import sys
import glob as _glob

import torch

MASK_ID = 156895
EOS_ID = 156892
PAD_ID = 156892

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def process_darc_gold_example(
    example,
    max_seq_len,
    block_size,
    *,
    mask_token_id=MASK_ID,
    pad_token_id=PAD_ID,
    eos_token_id=EOS_ID,
    mode="all_masked",                 # 'all_masked' (first trial) | 'ltr_reveal' | 'random'
    noise_range=(1.0, 1.0),            # used by ltr_reveal/random; (1,1) == all masked
    append_eos=True,                   # add one EOS target so the head learns to stop (gold_ids exclude it)
    progress_state=None,
    sigma_gate=0.0,
    source_name=None,                  # accepted (dataset.py passes it); unused
):
    """One collected record -> [ {input_ids, noisy_input_ids, attention_mask, labels, prompt_len} ].
    Returns a 1-element list to match the DMax MappingDataset flat-map convention."""
    prompt_ids = list(example["prompt_ids"])
    gold_ids = list(example["gold_ids"])
    P = int(example.get("prompt_len", len(prompt_ids)))

    seq = prompt_ids + gold_ids + ([eos_token_id] if append_eos else [])
    gen_end = len(seq)                                     # exclusive end of the real (prompt+gold+eos) region
    seq = seq[:max_seq_len]                                # truncate over-long
    if len(seq) < max_seq_len:                             # pad to fixed max_seq_len (DMax padding="max_length")
        seq = seq + [pad_token_id] * (max_seq_len - len(seq))
    input_ids = torch.tensor(seq, dtype=torch.long)

    P = min(P, max_seq_len)
    gen_end = min(gen_end, max_seq_len)
    pos = torch.arange(max_seq_len)
    maskable_mask = (pos >= P) & (pos < gen_end)           # the generated region only (never prompt or pad)

    if mode == "all_masked":
        noisy_input_ids = torch.where(maskable_mask, torch.full_like(input_ids, mask_token_id), input_ids)
    elif mode == "ltr_reveal":
        from data_transform_dbet import block_left_to_right_reveal
        noisy_input_ids = block_left_to_right_reveal(
            input_ids.clone(), noise_range, maskable_mask, mask_token_id, block_size,
            progress_state=progress_state, sigma_gate=sigma_gate)
    elif mode == "random":
        from data_transform import sft_noise_transition
        noisy_input_ids = sft_noise_transition(
            input_ids.clone(), noise_range, maskable_mask, mask_token_id,
            progress_state=progress_state, sigma_gate=sigma_gate)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # LLaDA masked-token objective (== DMax): loss only where the noisy stream is MASK.
    loss_positions = noisy_input_ids == mask_token_id
    labels = input_ids.clone()
    labels[~loss_positions] = -100

    attention_mask = (pos < gen_end).long()                # 1 = real (prompt+gold+eos), 0 = padding

    return [{
        "input_ids": input_ids,
        "noisy_input_ids": noisy_input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_len": torch.tensor(P, dtype=torch.long),
    }]


# --------------------------------------------------------------------------------------------------------
# JSONL shards -> HF Dataset (so the existing build_local_dataset + MappingDataset(transform) plumbing works)
# --------------------------------------------------------------------------------------------------------
def iter_gold_records(path_or_glob):
    """Yield records from one JSONL file or a glob of shard files, de-duplicated by 'rank'."""
    paths = sorted(_glob.glob(path_or_glob)) if any(c in path_or_glob for c in "*?[") else [path_or_glob]
    seen = set()
    for p in paths:
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue                                # tolerate a truncated trailing line
                r = rec.get("rank")
                if r in seen:
                    continue
                seen.add(r)
                yield rec


def build_hf_dataset_from_shards(path_or_glob, out_dir=None, keep=("prompt_ids", "gold_ids", "prompt_len", "rank")):
    """Read gold shards into a datasets.Dataset (columns in `keep`); optionally save_to_disk(out_dir/train)
    so `dataset.build_local_dataset(out_dir, transform=process_darc_gold_example)` picks it up."""
    from datasets import Dataset
    rows = [{k: rec[k] for k in keep if k in rec} for rec in iter_gold_records(path_or_glob)]
    ds = Dataset.from_list(rows)
    if out_dir is not None:
        ds.save_to_disk(os.path.join(out_dir, "train"))
        print(f"[darc-data] saved {len(ds)} rows -> {os.path.join(out_dir, 'train')}")
    return ds


# --------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    L, BLK = 64, 16
    # fake record: prompt of 20 tokens, gold response of 25 tokens
    rec = {"prompt_ids": list(range(100, 120)), "gold_ids": list(range(200, 225)), "prompt_len": 20}
    P, G = 20, 25

    out = process_darc_gold_example(rec, max_seq_len=L, block_size=BLK, mode="all_masked")[0]
    ii, ni, am, lab = out["input_ids"], out["noisy_input_ids"], out["attention_mask"], out["labels"]
    gen_end = P + G + 1                                     # +1 appended EOS
    assert ii.shape == ni.shape == am.shape == lab.shape == (L,)
    # prompt: clean, no loss
    assert (ni[:P] == ii[:P]).all() and (lab[:P] == -100).all()
    # generated region (incl EOS): all masked in noisy, labels == gold
    assert (ni[P:gen_end] == MASK_ID).all(), "all_masked must mask the whole generated region"
    assert (lab[P:gen_end] == ii[P:gen_end]).all() and (ii[gen_end - 1] == EOS_ID)
    # padding: clean, no loss, attention 0
    assert (ii[gen_end:] == PAD_ID).all() and (lab[gen_end:] == -100).all()
    assert (am[:gen_end] == 1).all() and (am[gen_end:] == 0).all()
    n_sup = int((lab != -100).sum())
    print(f"[all_masked] OK  supervised={n_sup} (== gen incl EOS = {G+1})  seq={L} blk={BLK}")
    assert n_sup == G + 1

    # ltr_reveal: reveals a left prefix per block, masks the rest -> fewer supervised than all_masked
    out2 = process_darc_gold_example(rec, max_seq_len=L, block_size=BLK, mode="ltr_reveal",
                                     noise_range=(0.5, 0.5))[0]
    n_sup2 = int((out2["labels"] != -100).sum())
    assert 0 < n_sup2 < n_sup, f"ltr_reveal should supervise a strict subset (got {n_sup2} vs {n_sup})"
    print(f"[ltr_reveal σ=0.5] OK  supervised={n_sup2} (< {n_sup})")
    print("data_transform_darc self-test PASSED")
