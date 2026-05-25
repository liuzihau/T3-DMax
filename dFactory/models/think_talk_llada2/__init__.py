# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# Module entry point. VeOmni's `ModelRegistry.register_modeling_path("models.think_talk_llada2")`
# discovers the class exported as `ModelClass` here.

from .configuration_think_talk_llada2 import ThinkTalkLLaDA2Config
from .modeling_think_talk_llada2 import ThinkTalkLLaDA2ForCausalLM

# VeOmni convention -- the registry imports `ModelClass` from this package.
ModelClass = ThinkTalkLLaDA2ForCausalLM

__all__ = [
    "ThinkTalkLLaDA2Config",
    "ThinkTalkLLaDA2ForCausalLM",
    "ModelClass",
]
