# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# Re-exports the registered Config class. Identical to
# ../think_talk_llada2_mini/configuration_think_talk_llada2.py -- see that file for the
# rationale (single source of truth via PYTHONPATH-resolved import). The only difference
# between this model_config dir and its sibling is `train_think: false` in config.json.

from models.think_talk_llada2.configuration_think_talk_llada2 import (  # noqa: F401
    LLaDA2MoeConfig,
    ThinkTalkLLaDA2Config,
)

__all__ = ["LLaDA2MoeConfig", "ThinkTalkLLaDA2Config"]
