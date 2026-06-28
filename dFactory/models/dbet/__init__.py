# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# Module entry point. VeOmni's `ModelRegistry.register_modeling_path("models.dbet")` discovers the class
# exported as `ModelClass` here (same convention as think_talk_llada2).

from .configuration_dbet import DbetConfig
from .modeling_dbet import DbetForDraftDecoding

# VeOmni convention -- the registry imports `ModelClass` from this package.
ModelClass = DbetForDraftDecoding

__all__ = [
    "DbetConfig",
    "DbetForDraftDecoding",
    "ModelClass",
]
