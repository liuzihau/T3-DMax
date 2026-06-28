# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
"""Build the INITIAL DBet checkpoint (run ONCE, off the training loop) so the VeOmni trainer can load it the
normal way (config_path == model_path == the output dir).

Why: `build_foundation_model` builds the model on `meta` then loads `model_path` whose keys must match the
model (`heavy.*` + `draft.*`). The raw DMax checkpoint has neither the `heavy.` prefix nor the drafter params,
so we assemble the initial DBet checkpoint here:
  1) DbetConfig from the heavy's config + the drafter overrides;
  2) DbetForDraftDecoding (random heavy + drafter);
  3) load the FROZEN DMax weights in-place into `model.heavy` (keeps the drafter's frozen embed/lm_head/norm
     refs valid, since they point INTO model.heavy);
  4) warm-start the drafter from the (now real) heavy bottom layers + zero-init Δh;
  5) save_pretrained -> config.json (model_type "dbet", architectures [DbetForDraftDecoding]) + weights.

Needs enough RAM/VRAM for ~2x the heavy briefly (random heavy + the loaded DMax state dict). DGX CPU is fine.

    PYTHONPATH=$(pwd)/VeOmni:$(pwd):$PYTHONPATH python scripts/build_dbet_init.py \
        --heavy_path /path/to/DMax-Math-16B-moe-merge --out_dir ./dbet_init \
        --draft_num_layers 5 --sel_layers 1,10,19 --block_size 32
"""

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))               # .../dFactory/scripts
_DFACTORY = os.path.abspath(os.path.join(_HERE, ".."))          # .../dFactory
for _p in (_DFACTORY, os.path.join(_DFACTORY, "VeOmni")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoConfig, AutoModelForCausalLM        # noqa: E402
from models.dbet import DbetConfig, DbetForDraftDecoding         # noqa: E402
from models.llada2_moe.modeling_llada2_moe import LLaDA2MoeModelLM    # noqa: E402
from models.llada2_moe.configuration_llada2_moe import LLaDA2MoeConfig  # noqa: E402

# make `model_type: "dbet"` resolvable by AutoConfig/AutoModel (mirrors train_dbet.py)
AutoConfig.register("dbet", DbetConfig)
AutoModelForCausalLM.register(DbetConfig, DbetForDraftDecoding)


def main():
    p = argparse.ArgumentParser(description="Assemble the initial DBet checkpoint from a frozen DMax heavy.")
    p.add_argument("--heavy_path", required=True, help="DMax-Math-16B checkpoint (the frozen heavy).")
    p.add_argument("--out_dir", required=True, help="output dir; use as BOTH config_path and model_path.")
    p.add_argument("--draft_num_layers", type=int, default=5)
    p.add_argument("--sel_layers", default="1,10,19")
    p.add_argument("--per_layer_prefix_fuse", action="store_true", default=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    # 1) DBet config = heavy config fields + drafter overrides (model_type stays the class default "dbet")
    heavy_cfg = LLaDA2MoeConfig.from_pretrained(args.heavy_path, trust_remote_code=True)
    hd = heavy_cfg.to_dict()
    for k in ("model_type", "architectures", "auto_map", "_name_or_path", "transformers_version"):
        hd.pop(k, None)
    cfg = DbetConfig(
        **hd,
        draft_num_layers=args.draft_num_layers, sel_layers=args.sel_layers,
        per_layer_prefix_fuse=args.per_layer_prefix_fuse,
        warmstart_from_heavy_bottom=False,          # we warm-start MANUALLY below; saved config stays False so
        heavy_path=args.heavy_path,                  # the trainer doesn't re-warmstart on meta at load time
    )
    print(f"[build_dbet_init] DbetConfig: hidden={cfg.hidden_size} heavy_layers={cfg.num_hidden_layers} "
          f"draft_layers={cfg.draft_num_layers} sel={cfg.sel_layers_list} m={cfg.m}")

    # 2) build model (random heavy + drafter); warmstart auto-call skipped (flag False)
    model = DbetForDraftDecoding(cfg).to(device=args.device, dtype=torch.bfloat16)

    # 3) load the frozen DMax heavy weights IN PLACE (keeps draft.frozen_* refs valid)
    dmax = LLaDA2MoeModelLM.from_pretrained(args.heavy_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    missing, unexpected = model.heavy.load_state_dict(dmax.state_dict(), strict=False)
    print(f"[build_dbet_init] heavy load: {len(missing)} missing, {len(unexpected)} unexpected (expect ~0)")
    del dmax

    # 4) warm-start drafter from the real heavy bottom + zero Δh; freeze
    model.init_draft_layers_warmstart()
    model._apply_freeze_flags()

    # 5) save self-contained checkpoint (config.json + weights). config_path == model_path == out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    model.save_pretrained(args.out_dir, safe_serialization=True)
    n_train = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f"[build_dbet_init] saved -> {args.out_dir}  (trainable drafter params = {n_train:,})")
    print(f"[build_dbet_init] set the yaml's model.config_path AND model.model_path to {args.out_dir}; "
          f"model.tokenizer_path to {args.heavy_path}")


if __name__ == "__main__":
    main()
