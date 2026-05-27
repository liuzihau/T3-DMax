# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# Diagnostic: inspect a saved T3-D checkpoint to distinguish between three hypotheses
# about why eval CE is flat-and-bad across sigma:
#
#   (A) Early training: cross-attn weights still near depth-scaled init, talk hasn't
#       learned to fire its anchor pathway yet. Fix: wait, no architecture change.
#
#   (B) lm_head drift: the trainable lm_head (initialised from LLaDA's tied weight)
#       has moved far from the reference, destroying the vocabulary calibration that
#       lets the anchor's logits decode to correct tokens. Fix: restart with Strategy C
#       (lr_lm_head_ratio: 0.02) to slow lm_head's drift.
#
#   (C) Both: cross-attn dead AND lm_head drifted. Fix: both of the above.
#
# Output: a small table + verdict.
#
# Usage:
#   PYTHONPATH=$(pwd)/VeOmni:$(pwd):$PYTHONPATH \
#   python tasks/diagnose_t3d_weights.py \
#     --t3d_ckpt ./outputs/<run>/checkpoints/global_step_15000/hf_ckpt \
#     --llada_ckpt ../LLaDA2.0-mini-moe-merge

import argparse
import glob
import os
import sys
from typing import Dict, Optional

import torch
from safetensors.torch import load_file


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--t3d_ckpt", required=True,
                   help="Path to T3-D hf_ckpt directory (contains *.safetensors)")
    p.add_argument("--llada_ckpt", required=True,
                   help="Path to reference LLaDA-2.0-mini directory (contains *.safetensors)")
    p.add_argument("--talk_num_layers", type=int, default=4,
                   help="Number of talk layers in the T3-D model (for layout)")
    return p.parse_args()


def load_state(directory: str) -> Dict[str, torch.Tensor]:
    """Load all safetensors shards under `directory` into one state dict."""
    shards = sorted(glob.glob(os.path.join(directory, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(
            f"No .safetensors shards found in {directory!r}. "
            f"Listing: {os.listdir(directory) if os.path.isdir(directory) else '(no dir)'}"
        )
    state: Dict[str, torch.Tensor] = {}
    for s in shards:
        state.update(load_file(s))
    return state


def depth_scaled_init_std(layer_idx: int, base: float = 0.02) -> float:
    """The depth-scaled init std used in init_talk_layers_depth_scaled() for output projs."""
    return base / (2.0 * (layer_idx + 1)) ** 0.5


def main():
    args = parse_args()

    print(f"[diagnose] Loading T3-D state from {args.t3d_ckpt!r} ...")
    t3d_state = load_state(args.t3d_ckpt)
    print(f"[diagnose] Loaded {len(t3d_state)} tensors from T3-D.")

    print(f"[diagnose] Loading reference LLaDA state from {args.llada_ckpt!r} ...")
    ref_state = load_state(args.llada_ckpt)
    print(f"[diagnose] Loaded {len(ref_state)} tensors from LLaDA.\n")

    # ---- (1) Cross-attn dense projection weights ------------------------------
    # init_talk_layers_depth_scaled() sets output projections (`dense`, `down_proj`)
    # to std = initializer_range / sqrt(2 * (layer_idx + 1)). For the cross-attn
    # output projection (`cross_attention.dense`), that's the relevant init scale.
    # If cross-attn is dead (still at init), std should match the formula.
    print("=" * 78)
    print("  (1) Cross-attn output projection weight stats")
    print("=" * 78)
    print(f"  {'layer':>5}  {'cur_std':>10}  {'init_std':>10}  {'ratio':>8}  {'cur_norm':>10}")
    cross_attn_ratios = []
    for layer in range(args.talk_num_layers):
        key = f"talk_model.layers.{layer}.cross_attention.dense.weight"
        if key not in t3d_state:
            print(f"  L{layer:>3}: MISSING key {key!r}")
            continue
        w = t3d_state[key].float().cpu()
        cur_std = w.std().item()
        cur_norm = w.norm().item()
        init = depth_scaled_init_std(layer)
        ratio = cur_std / init
        cross_attn_ratios.append(ratio)
        print(f"  {layer:>5}  {cur_std:>10.5f}  {init:>10.5f}  {ratio:>8.2f}  {cur_norm:>10.3f}")

    cross_attn_alive = (
        any(r > 2.5 for r in cross_attn_ratios) if cross_attn_ratios else False
    )
    print(
        "\n  Reading: ratio ~1x = still at init (cross-attn dead). "
        "ratio > 2.5 = actively learning."
    )

    # ---- (2) Layer-0 anchor_conditioning gate ---------------------------------
    # If gate is fixed (learnable=false), it lives in the buffer 'fixed_gate' and
    # shouldn't appear in safetensors (buffer is persistent=False). If learnable,
    # it'd be `talk_model.layers.0.anchor_conditioning.alpha`.
    print("\n" + "=" * 78)
    print("  (2) Layer-0 anchor_conditioning gate")
    print("=" * 78)
    alpha_key = "talk_model.layers.0.anchor_conditioning.alpha"
    if alpha_key in t3d_state:
        alpha = float(t3d_state[alpha_key].item())
        gate = torch.sigmoid(torch.tensor(alpha)).item()
        print(f"  learnable: alpha={alpha:.4f}  sigmoid(alpha)={gate:.4f}")
    else:
        # Look for the anchor_norm to confirm the module exists, then note that the
        # gate is a buffer not in state_dict (so the yaml value 0.2 is in effect).
        anchor_norm_key = "talk_model.layers.0.anchor_conditioning.anchor_norm.weight"
        if anchor_norm_key in t3d_state:
            an = t3d_state[anchor_norm_key].float()
            print(
                f"  fixed-gate mode (gate value from yaml; not in state_dict). "
                f"anchor_norm.weight std={an.std().item():.4f}  norm={an.norm().item():.3f}"
            )
        else:
            print(f"  WARNING: neither {alpha_key!r} nor anchor_norm weight found. "
                  f"Layer-0 anchor injection module may not have been built.")

    # ---- (3) lm_head drift from LLaDA tied weight -----------------------------
    print("\n" + "=" * 78)
    print("  (3) lm_head drift from LLaDA tied weight")
    print("=" * 78)
    if "lm_head.weight" not in t3d_state:
        print("  MISSING t3d_state['lm_head.weight']")
        lm_drift = None
    else:
        cur = t3d_state["lm_head.weight"].float().cpu()
        # LLaDA-2.0 ties lm_head to word_embeddings. The naming varies.
        ref: Optional[torch.Tensor] = None
        for ref_key in (
            "model.word_embeddings.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",  # in case the conversion saved it explicitly
        ):
            if ref_key in ref_state:
                ref = ref_state[ref_key].float().cpu()
                print(f"  reference key: {ref_key!r}")
                break
        if ref is None:
            print(
                "  WARNING: no recognised reference key in LLaDA state_dict. "
                f"Available top-keys: {list(ref_state.keys())[:6]}..."
            )
            lm_drift = None
        else:
            if cur.shape != ref.shape:
                print(
                    f"  WARNING: shape mismatch cur={tuple(cur.shape)} "
                    f"ref={tuple(ref.shape)}. Skipping drift calc."
                )
                lm_drift = None
            else:
                diff = cur - ref
                lm_drift = diff.norm().item() / ref.norm().item()
                cos_sim = (cur * ref).sum().item() / (
                    cur.norm().item() * ref.norm().item() + 1e-12
                )
                print(f"  lm_head.weight  current: norm={cur.norm().item():.3f}  "
                      f"std={cur.std().item():.5f}")
                print(f"  reference                norm={ref.norm().item():.3f}  "
                      f"std={ref.std().item():.5f}")
                print(f"  drift  = ||cur - ref|| / ||ref|| = {lm_drift:.4f}")
                print(f"  cosine similarity              = {cos_sim:.4f}")

    print(
        "\n  Reading: drift<0.02 = barely moved (early training). "
        "0.05-0.15 = adapting. >0.20 = significant drift (concerning)."
    )

    # ---- (4) Verdict ----------------------------------------------------------
    print("\n" + "=" * 78)
    print("  Verdict")
    print("=" * 78)

    if lm_drift is None:
        print("  (could not compute lm_head drift; check reference path)")
    else:
        if lm_drift < 0.02 and not cross_attn_alive:
            print(
                "  [A] Early training: cross-attn at init AND lm_head barely moved. "
                "Talk pathway has not learned. Suggest waiting for more steps OR "
                "investigating why gradient flow is so weak."
            )
        elif lm_drift > 0.20 and not cross_attn_alive:
            print(
                "  [B] lm_head drift dominant: cross-attn dead AND lm_head moved a lot. "
                "Most of the training-loss reduction (7.1 -> 4.5) probably came from "
                "lm_head over-fitting to a flat output prior, not from talk learning. "
                "Fix: restart with lr_lm_head_ratio: 0.02 (Strategy C, already coded)."
            )
        elif lm_drift > 0.20 and cross_attn_alive:
            print(
                "  [C] Both pathways active but failing. Talk is learning, lm_head is drifting, "
                "but the combination doesn't unlock useful predictions. Architectural rethink "
                "needed (capacity? gate too small? cross-attn target wrong?)."
            )
        elif lm_drift < 0.05 and cross_attn_alive:
            print(
                "  Cross-attn learning, lm_head close to LLaDA. Probably healthy early state; "
                "wait for more training and re-eval."
            )
        else:
            print(
                "  Mixed signals. Inspect numbers above directly; verdict heuristics didn't "
                "match a clean pattern."
            )

    print("=" * 78)


if __name__ == "__main__":
    main()
