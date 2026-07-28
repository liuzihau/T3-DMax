# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# DARC head config. Kept a plain dataclass for the first trial (the head module is standalone-testable);
# upgrade to a transformers.PretrainedConfig at base-model integration time (like DbetConfig) if needed.
# Defaults match the DMax-Math-16B / LLaDA2-mini-moe backbone so the head warm-starts on the right shapes.

from dataclasses import dataclass


@dataclass
class DarcConfig:
    # --- backbone dims (LLaDA2-mini-moe) ---
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_key_value_heads: int = 4          # GQA
    head_dim: int = 128
    intermediate_size: int = 5120         # DARC MLP (dense SwiGLU)
    vocab_size: int = 157184
    rms_norm_eps: float = 1e-6
    rotary_dim: int = 64                  # partial rotary (partial_rotary_factor 0.5 * head_dim 128)
    rope_theta: float = 600000.0

    # --- DARC specifics ---
    tap_hidden_index: int = 18            # hidden_states index to tap == the probe-plot "L{i}" label.
                                          # hs[i] = output of decoder layer (i-1); hs[0]=embeddings, hs[20]=final.
                                          # So tap_hidden_index=18 -> decoder layer 17 out (plot "L18").
    block_size: int = 32                  # diffusion block; AR chain resets at each block boundary
    top_k: int = 5                        # soft-embed top-k (start k=5)
    soft_tau: float = 1.0                 # soft-embed temperature
    hidden_act: str = "silu"

    # --- training / freeze ---
    freeze_backbone: bool = True          # freeze everything but attn+mlp+fuse (+optional LoRA later)
    use_fuse: bool = True                 # build the Loss-2 fuse (set False for Loss-1-only trials -> ~50M)
    loss1_weight: float = 1.0             # per-position AR readout CE
    loss2_weight: float = 1.0             # fused -> L19+ CE (added later)
    attn_impl: str = "sdpa"
