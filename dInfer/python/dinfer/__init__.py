#!/usr/bin/python
#****************************************************************#
# ScriptName: python/llada/__init__.py
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2025-09-15 19:48
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2025-09-15 19:48
# Function: 
#***************************************************************#

__version__ = "0.1"


# T3-D MODIFIED: DMax's original __init__ eagerly imports vllm/sglang-backed
# decoders. Those are unavailable in T3-D's single-GPU training env. Wrap each
# group in try/except so `import dinfer.decoding.generate_t3d` works without
# pulling in the full DMax inference stack.
try:
    from .decoding.parallel_strategy import (
        ThresholdParallelDecoder,
        CreditThresholdParallelDecoder,
        HierarchyDecoder,
    )
except ImportError:
    pass

try:
    from .decoding.generate_uniform import (
        DiffusionLLM,
        BlockWiseDiffusionLLM,
        VicinityCacheDiffusionLLM,
        BlockWiseDiffusionLLMWithSP,
        BlockDiffusionLLMAttnmask,
        BlockDiffusionLLM,
        IterSmoothDiffusionLLM,
        IterSmoothWithVicinityCacheDiffusionLLM,
    )
except ImportError:
    pass

try:
    from .decoding.serving import DiffusionLLMServing, SamplingParams
except ImportError:
    pass

try:
    from .decoding.utils import BlockIteratorFactory, KVCacheFactory
except ImportError:
    pass
