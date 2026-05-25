# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# Re-exports the registered Config class. Identical to the other model_config dirs in
# this folder -- see ../think_talk_llada2_mini/configuration_think_talk_llada2.py for
# the rationale. The only difference between sibling model_config dirs is which fields
# config.json sets (train_think, train_lm_head, etc.).

from models.think_talk_llada2.configuration_think_talk_llada2 import (  # noqa: F401
    LLaDA2MoeConfig,
    ThinkTalkLLaDA2Config,
)

__all__ = ["LLaDA2MoeConfig", "ThinkTalkLLaDA2Config"]
