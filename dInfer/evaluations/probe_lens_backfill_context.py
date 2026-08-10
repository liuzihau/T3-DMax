# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# CPU-ONLY CONTEXT BACKFILL for lens-surface runs recorded before probe_lens_surface.py wrote the
# per-sample context json. No GPU, no model weights -- only the tokenizer and the shards.
#
# How it works
#   prompt : re-tokenize the same task rows the probe consumed (same task/limit/order, same chat
#            template) -> prompt_ids, P. Deterministic, so it reproduces exactly what the probe fed.
#   gen    : the generated tokens are ALREADY in the shards -- t_c per (block, position) -- so the
#            generated prefix is reassembled by ordering t_c by (block, pos).
#   check  : block 0's first recorded position must equal P - first_block_start; mismatch means the
#            task/limit/tokenizer do not match the run, and the script refuses to write.
#
# Run (CPU studio):
#   python probe_lens_backfill_context.py --run_dir runs/lens_smoke \
#       --tokenizer_path ../../DMax-16B --task gsm8k --limit 4

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from eval_tasks import load_task  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Backfill the per-sample context json (CPU only)")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--tokenizer_path", required=True, help="dir with tokenizer.json (original download OK)")
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k", "math500", "algebra", "asdiv"])
    ap.add_argument("--limit", type=int, default=None, help="the probe's --limit (default: from meta)")
    ap.add_argument("--block_length", type=int, default=None, help="default: from meta")
    ap.add_argument("--gt_jsonl_path", default=None)
    ap.add_argument("--force", action="store_true", help="write even if the position self-check fails")
    args = ap.parse_args()

    meta_path = os.path.join(args.run_dir, "probe_lens_surface_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    limit = args.limit or meta.get("n_examples")
    block_length = args.block_length or meta.get("block_length", 32)
    task = args.task or meta.get("task", "gsm8k")
    if not limit:
        raise SystemExit("cannot determine --limit (no meta json); pass it explicitly")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    rows = load_task(task, limit=limit, gt_jsonl_path=args.gt_jsonl_path)
    print(f"[backfill] task={task} limit={limit} block_length={block_length} rows={len(rows)}")

    # gather (block, pos, t_c) per sample from the shards
    per_sample = {}
    for p in sorted(glob.glob(os.path.join(args.run_dir, "probe_lens_surface_shard*.npz"))):
        z = np.load(p)
        s, b, pos, tc = z["sample"], z["block"], z["pos"], z["t_c"]
        for si in np.unique(s):
            m = s == si
            key = np.stack([b[m].astype(np.int64), pos[m].astype(np.int64)], 1)
            uniq, idx = np.unique(key, axis=0, return_index=True)
            d = per_sample.setdefault(int(si), {})
            for (bb, pp), tt in zip(uniq, tc[m][idx]):
                d[(int(bb), int(pp))] = int(tt)

    contexts, bad = [], 0
    for si in sorted(per_sample):
        if si >= len(rows):
            print(f"[backfill] sample {si} beyond the task rows -- skipped")
            continue
        msgs = [{"role": "user", "content": rows[si]["prompt"]}]
        prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if isinstance(prompt_ids[0], (list, tuple)):
            prompt_ids = list(prompt_ids[0])
        P = len(prompt_ids)
        fbs = (P // block_length) * block_length
        cells = per_sample[si]
        first_pos_b0 = min((pp for (bb, pp) in cells if bb == 0), default=None)
        expected = P - fbs
        ok = first_pos_b0 is None or first_pos_b0 == expected
        if not ok:
            bad += 1
            print(f"[backfill] sample {si}: SELF-CHECK FAILED -- block-0 first position is {first_pos_b0}, "
                  f"expected {expected} (P={P}). Task/limit/tokenizer likely differ from the run.")
        gen_ids = [cells[k] for k in sorted(cells)]
        contexts.append(dict(sample=si, P=P, block_length=block_length,
                             eos_cut=P + len(gen_ids) - 1, first_block_start=fbs,
                             prompt_ids=list(map(int, prompt_ids)), gen_ids=gen_ids,
                             backfilled=True, self_check_ok=bool(ok)))

    if bad and not args.force:
        raise SystemExit(f"[backfill] {bad} sample(s) failed the self-check; nothing written. "
                         f"Check --task/--limit/--tokenizer_path, or pass --force.")
    out = os.path.join(args.run_dir, "probe_lens_context_shard000.json")
    with open(out, "w") as fh:
        json.dump(contexts, fh)
    print(f"[backfill] wrote {out} for {len(contexts)} sample(s) "
          f"({'all self-checks passed' if not bad else f'{bad} forced'})")


if __name__ == "__main__":
    main()
