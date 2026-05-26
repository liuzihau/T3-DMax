# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# Re-exports the registered Config class. Same trick as the other model_config dirs --
# see ../think_talk_llada2_mini/configuration_think_talk_llada2.py for the rationale.
# Only config.json differs between sibling dirs.

from models.think_talk_llada2.configuration_think_talk_llada2 import (  # noqa: F401
    LLaDA2MoeConfig,
    ThinkTalkLLaDA2Config,
)

__all__ = ["LLaDA2MoeConfig", "ThinkTalkLLaDA2Config"]
