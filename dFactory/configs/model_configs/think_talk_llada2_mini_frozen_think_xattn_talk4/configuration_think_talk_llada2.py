# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# Re-exports the registered Config class. Identical to
# ../think_talk_llada2_mini/configuration_think_talk_llada2.py -- see that file for the
# rationale (single source of truth via PYTHONPATH-resolved import). The only differences
# vs the sibling frozen_think config.json are:
#   - talk_num_layers: 4 (was 2)
#   - anchor_injection_mode: "hybrid_xattn" (gated residual at layer 0 + per-layer cross-attn)
#   - frozen think (train_think=false), trainable talk + lm_head.

from models.think_talk_llada2.configuration_think_talk_llada2 import (  # noqa: F401
    LLaDA2MoeConfig,
    ThinkTalkLLaDA2Config,
)

__all__ = ["LLaDA2MoeConfig", "ThinkTalkLLaDA2Config"]
