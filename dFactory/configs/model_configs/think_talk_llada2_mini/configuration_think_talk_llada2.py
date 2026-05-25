# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# Re-exports the registered Config class so HuggingFace `trust_remote_code` loading
# (driven by `auto_map` in config.json) and the VeOmni-registered class are the same
# class. This eliminates the drift trap where field defaults or helper methods get
# out of sync between two parallel definitions.
#
# Why this works inside HF's `trust_remote_code` flow:
#   - HF dynamically imports this file via importlib (path-based, not package-relative).
#   - The training script puts `dFactory/` on PYTHONPATH, so the absolute import below
#     resolves cleanly to the registered Config class.
#   - The training-time path is the only path we care about; T3-D is not built to be
#     loaded standalone from HF Hub.
#
# Pattern divergence from DMax: DMax keeps a duplicated standalone LLaDA2MoeConfig
# in its model_configs/llada2_mini/. We collapse the duplicate instead. If DMax-style
# standalone loading is ever needed, copy the full Config class definition from
# `models/think_talk_llada2/configuration_think_talk_llada2.py` into this file.

from models.think_talk_llada2.configuration_think_talk_llada2 import (  # noqa: F401
    LLaDA2MoeConfig,
    ThinkTalkLLaDA2Config,
)

__all__ = ["LLaDA2MoeConfig", "ThinkTalkLLaDA2Config"]
