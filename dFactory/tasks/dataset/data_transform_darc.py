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
    mode="reveal_prior",               # 'reveal_prior' (first trial) | 'all_masked' | 'ltr_reveal' | 'random'
    noise_range=(1.0, 1.0),            # used by ltr_reveal/random; (1,1) == all masked
    append_eos=True,                   # add one EOS target so the head learns to stop (gold_ids exclude it)
    progress_state=None,
    sigma_gate=0.0,
    source_name=None,                  # accepted (dataset.py passes it); unused
):
    """One collected record -> LIST of training instances {input_ids, noisy_input_ids, attention_mask,
    labels, prompt_len, active_bs} (list matches the DMax MappingDataset flat-map convention).

    'reveal_prior' (VALIDATED first-trial setup, see dFactory/scripts/smoke_darc_integration.py): emit ONE
    instance per generated block b -- reveal [0,bs) as committed gold (hard tokens), mask [bs, L) (block b +
    everything after), loss on block b only. This matches the model's inference-time forward-1 (prior blocks
    committed, current block masked); masking the WHOLE generated region instead collapses the tap signal
    (recall avg 0.44 -> 0.12) and must not be used for training. The other modes emit ONE instance
    (all-masked / left-to-right reveal / random) and are for reference/ablation only."""
    prompt_ids = list(example["prompt_ids"])
    gold_ids = list(example["gold_ids"])
    P = int(example.get("prompt_len", len(prompt_ids)))

    seq = prompt_ids + gold_ids + ([eos_token_id] if append_eos else [])
    gen_end = min(len(seq), max_seq_len)                   # exclusive end of the real (prompt+gold+eos) region
    seq = seq[:max_seq_len]                                # truncate over-long
    if len(seq) < max_seq_len:                             # pad to fixed max_seq_len (DMax padding="max_length")
        seq = seq + [pad_token_id] * (max_seq_len - len(seq))
    input_ids = torch.tensor(seq, dtype=torch.long)
    P = min(P, max_seq_len)
    pos = torch.arange(max_seq_len)
    attention_mask = (pos < gen_end).long()                # 1 = real (prompt+gold+eos), 0 = padding
    B = block_size

    def _inst(noisy, labels, active_bs=-1):
        return {"input_ids": input_ids, "noisy_input_ids": noisy, "attention_mask": attention_mask,
                "labels": labels, "prompt_len": torch.tensor(P, dtype=torch.long),
                "active_bs": torch.tensor(active_bs, dtype=torch.long)}

    if mode == "reveal_prior":
        out = []
        first = ((P + B - 1) // B) * B                     # first full-block boundary at/after the prompt
        for bs in range(first, gen_end, B):                # one instance per generated block
            be = min(bs + B, gen_end)
            noisy = input_ids.clone()
            noisy[bs:] = mask_token_id                     # reveal [0,bs) committed gold; mask block b + future
            labels = torch.full_like(input_ids, -100)
            labels[bs:be] = input_ids[bs:be]               # loss on block b (head skips the rel-pos-0 seed)
            out.append(_inst(noisy, labels, active_bs=bs))
        return out

    # ---- single-instance reference/ablation modes ----
    maskable = (pos >= P) & (pos < gen_end)                # generated region only (never prompt or pad)
    if mode == "all_masked":
        noisy = torch.where(maskable, torch.full_like(input_ids, mask_token_id), input_ids)
    elif mode == "ltr_reveal":
        from data_transform_dbet import block_left_to_right_reveal
        noisy = block_left_to_right_reveal(input_ids.clone(), noise_range, maskable, mask_token_id, B,
                                           progress_state=progress_state, sigma_gate=sigma_gate)
    elif mode == "random":
        from data_transform import sft_noise_transition
        noisy = sft_noise_transition(input_ids.clone(), noise_range, maskable, mask_token_id,
                                     progress_state=progress_state, sigma_gate=sigma_gate)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    labels = input_ids.clone()
    labels[noisy != mask_token_id] = -100                  # LLaDA masked-token objective (loss only at MASK)
    return [_inst(noisy, labels)]


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

    # --- reveal_prior (first-trial default): prompt 8, gold 40 -> blocks at bs=16,32,48 ---
    rec = {"prompt_ids": list(range(100, 108)), "gold_ids": list(range(200, 240)), "prompt_len": 8}
    P, gen_end = 8, 8 + 40 + 1                              # +1 appended EOS
    insts = process_darc_gold_example(rec, max_seq_len=L, block_size=BLK, mode="reveal_prior")
    bss = [int(x["active_bs"]) for x in insts]
    assert bss == [16, 32, 48], bss                        # one instance per generated block
    for inst in insts:
        ii, ni, lab, bs = inst["input_ids"], inst["noisy_input_ids"], inst["labels"], int(inst["active_bs"])
        be = min(bs + BLK, gen_end)
        assert (ni[:bs] == ii[:bs]).all(), "reveal_prior: [0,bs) must be committed gold (revealed)"
        assert (ni[bs:] == MASK_ID).all(), "reveal_prior: block b + future must be masked"
        assert (lab[:bs] == -100).all() and (lab[be:] == -100).all()
        assert (lab[bs:be] == ii[bs:be]).all(), "loss targets == gold on block b"
    n_sup = sum(int((x["labels"] != -100).sum()) for x in insts)
    print(f"[reveal_prior] OK  {len(insts)} block-instances bs={bss}  total_supervised={n_sup}")

    # --- all_masked (reference): whole generated region masked, single instance ---
    out = process_darc_gold_example(rec, max_seq_len=L, block_size=BLK, mode="all_masked")[0]
    ni, lab = out["noisy_input_ids"], out["labels"]
    assert (ni[P:gen_end] == MASK_ID).all() and (lab[P:gen_end] == out["input_ids"][P:gen_end]).all()
    assert (lab[:P] == -100).all() and (lab[gen_end:] == -100).all()
    print(f"[all_masked] OK  supervised={int((lab != -100).sum())} (reference mode)")

    # --- ltr_reveal (reference): reveals a left prefix per block -> strict subset ---
    out2 = process_darc_gold_example(rec, max_seq_len=L, block_size=BLK, mode="ltr_reveal",
                                     noise_range=(0.5, 0.5))[0]
    n2 = int((out2["labels"] != -100).sum())
    assert 0 < n2 < int((lab != -100).sum())
    print(f"[ltr_reveal σ=0.5] OK  supervised={n2}")
    print("data_transform_darc self-test PASSED")
