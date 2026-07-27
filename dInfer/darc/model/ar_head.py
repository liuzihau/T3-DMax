"""DARC head: AR refinement chain grafted after layer 18 of a frozen DMax model.

TODO(first trial): implement
  - SoftEmbed(logits, embed, k): top-k softmax-weighted sum over FROZEN base input embedding.
  - DARCHead(nn.Module): 1 cross-attn (Q<-h_i, K/V<-s_{<i}) + 1 SwiGLU MLP + Fuse(2D->D, residual to h).
  - forward_parallel(h, gold_ids, causal_mask): produce detached soft-embeds, then one masked pass for Loss1.
  - forward_step(h, s_prefix): single AR step for sequential inference.
See ../README.md for the full design and train/inference-consistency notes.
"""
