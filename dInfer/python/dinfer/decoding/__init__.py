# T3-D MODIFIED: each group wrapped in try/except so importing just generate_t3d
# works without pulling DMax's vllm/sglang-backed decoders.
try:
    from .parallel_strategy import (
        ThresholdParallelDecoder,
        CreditThresholdParallelDecoder,
        HierarchyDecoder,
    )
except ImportError:
    pass

try:
    from .generate_uniform import (
        BlockWiseDiffusionLLM,
        VicinityCacheDiffusionLLM,
        IterSmoothWithVicinityCacheDiffusionLLM,
        BlockWiseDiffusionLLMWithSP,
        IterSmoothDiffusionLLM,
        BlockDiffusionLLMAttnmask,
        BlockDiffusionLLM,
    )
except ImportError:
    pass

try:
    from .utils import BlockIteratorFactory, KVCacheFactory, TokenArray
except ImportError:
    pass
