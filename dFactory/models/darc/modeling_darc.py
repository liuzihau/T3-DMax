# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# DARC head: AR refinement chain grafted after layer 18 of a frozen DMax (LLaDA2-MoE) model.
# Mirrors the dFactory/models/dbet/ layout (configuration_*.py + modeling_*.py).
#
# TODO(first trial): implement
#   - SoftEmbed(logits, embed, k): top-k softmax-weighted sum over the FROZEN base input embedding.
#   - DARCHead(nn.Module): 1 cross-attn (Q<-h_i, K/V<-s_{<i}, base RoPE/LN) + 1 SwiGLU MLP + Fuse(2D->D,
#     residual to h).
#   - forward_parallel(h, gold_ids, causal_mask): produce detached soft-embeds, then ONE causal-masked pass
#     for the per-position Loss 1 (positions i>=1; pos 0 is a frozen seed).
#   - forward_step(h_i, s_prefix): single AR step for sequential inference.
#
# See ./README.md for the full design, the two-loss training scheme, and the train/inference-consistency
# notes. Backbone dims (DMax-Math-16B / LLaDA2-mini-moe): D=2048, heads 16Q/4KV, head_dim 128, FFN 5120,
# V=157184, 20 layers; tap after layer 18.
