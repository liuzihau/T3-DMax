# Copyright 2026 University of Sydney. Apache-2.0.
"""Extract the fine-tuned HEAVY from a DBet fine-tune checkpoint into a plain LLaDA2Moe checkpoint the dInfer
sweep can load as `heavy_path`.

The fine-tune hf_ckpt is the FULL DBet model: keys `heavy.model.*` / `heavy.lm_head.*` / `draft.*` under a
DbetConfig. The sweep loads heavy_path as a standalone LLaDA2MoeModelLM, which expects `model.*` / `lm_head.*`
(no `heavy.` prefix). So: keep only `heavy.*`, strip the prefix, and drop in the original DMax config+tokenizer.

  python scripts/extract_heavy_from_ft.py \
      --ft_ckpt   ../dFactory/dmax_ft_outputs/checkpoints/global_step_40000/hf_ckpt \
      --dmax_ref  ../DMax-Math-16B-moe-merge \
      --out_dir   ../dFactory/dmax_ft_heavy_40000    [--dtype bf16] [--shard_gb 5]
"""
import argparse, glob, os, shutil

import torch
from safetensors.torch import load_file, save_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ft_ckpt", required=True, help="fine-tune hf_ckpt dir (heavy.* + draft.*)")
    p.add_argument("--dmax_ref", required=True, help="original DMax dir (source of LLaDA2Moe config.json + tokenizer)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "keep"], help="cast heavy weights (inference=bf16).")
    p.add_argument("--shard_gb", type=float, default=5.0, help="shard the output into ~N GB safetensors files.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.ft_ckpt, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {args.ft_ckpt}")

    heavy = {}
    n_total = 0
    for f in files:
        sd = load_file(f)
        n_total += len(sd)
        for k, v in sd.items():
            if k.startswith("heavy."):
                if args.dtype == "bf16" and v.is_floating_point():
                    v = v.to(torch.bfloat16)
                heavy[k[len("heavy."):]] = v            # heavy.model.layers.0 -> model.layers.0
    if not heavy:
        raise RuntimeError("no 'heavy.*' keys found -- is this a DBet fine-tune ckpt? (train_dbet drops heavy.*)")
    print(f"[extract] kept {len(heavy)} heavy tensors (stripped 'heavy.') from {n_total} total; dropped draft.*")

    # sharded save (mirrors HF: model-00001-of-000NN.safetensors + index.json)
    limit = int(args.shard_gb * 1e9)
    shards, cur, cur_bytes = [], {}, 0
    for k, v in heavy.items():
        b = v.numel() * v.element_size()
        if cur and cur_bytes + b > limit:
            shards.append(cur); cur, cur_bytes = {}, 0
        cur[k] = v; cur_bytes += b
    if cur:
        shards.append(cur)
    weight_map = {}
    if len(shards) == 1:
        name = "model.safetensors"
        save_file(shards[0], os.path.join(args.out_dir, name), metadata={"format": "pt"})
        weight_map = {k: name for k in shards[0]}
    else:
        for i, sh in enumerate(shards, 1):
            name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
            save_file(sh, os.path.join(args.out_dir, name), metadata={"format": "pt"})
            weight_map.update({k: name for k in sh})
    if len(shards) > 1:
        import json
        total = sum(v.numel() * v.element_size() for v in heavy.values())
        with open(os.path.join(args.out_dir, "model.safetensors.index.json"), "w") as fh:
            json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, fh, indent=2)
    print(f"[extract] wrote {len(shards)} shard(s) -> {args.out_dir}")

    # config + tokenizer from the ORIGINAL DMax (LLaDA2Moe config, not DbetConfig)
    copied = 0
    for fn in os.listdir(args.dmax_ref):
        if (fn.endswith(".json") and "index" not in fn) or "tokenizer" in fn or fn.endswith((".model", ".jinja")):
            shutil.copy(os.path.join(args.dmax_ref, fn), os.path.join(args.out_dir, fn))
            copied += 1
    print(f"[extract] copied {copied} config/tokenizer files from {args.dmax_ref}")
    print(f"[extract] DONE. Point the sweep's heavy= at: {args.out_dir}")


if __name__ == "__main__":
    main()
