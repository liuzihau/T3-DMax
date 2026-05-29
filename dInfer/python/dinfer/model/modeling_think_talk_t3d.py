# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# T3-D inference wrapper. Light shim over `ThinkTalkLLaDA2ForCausalLM`
# (defined in dFactory/models/think_talk_llada2/) that loads the model in eval
# mode and exposes the two primitives the block-decoding loop needs:
#
#   prefill_think(input_ids)              -> anchor_prompt, think_kv
#   extend_think(new_segment, think_kv)   -> anchor_new, think_kv'
#   forward_talk(block, anchor, ...)      -> logits_block
#
# Compared with DMax's dInfer/python/dinfer/model/modeling_llada2_moe.py, this
# shim is intentionally minimal:
#   - No vllm/sglang backend wiring (single-GPU only).
#   - Uses HuggingFace's DynamicCache rather than DMax's custom KVCache.
#   - No CUDA graphs / torch.compile (correctness-first).
#
# Once the loop is verified end-to-end, the optimisation stack (compile,
# cuda graphs, vllm) can be layered back in by mirroring DMax's wrapper.

import os
import sys
from typing import Optional, Tuple

import torch
from transformers import AutoTokenizer


def _ensure_dfactory_on_path():
    """`ThinkTalkLLaDA2ForCausalLM` lives under T3-DMax/dFactory/. dInfer is one
    level deeper, so we add dFactory/ to sys.path so its `models.*` and `tasks.*`
    imports resolve the same way the training script sees them.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    # here = .../T3-DMax/dInfer/python/dinfer/model
    # dFactory = .../T3-DMax/dFactory
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    dfactory = os.path.join(repo_root, "dFactory")
    veomni = os.path.join(dfactory, "VeOmni")
    for p in (dfactory, veomni):
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_dfactory_on_path()

# These imports must come after the path patch.
from models.think_talk_llada2.configuration_think_talk_llada2 import (  # noqa: E402
    ThinkTalkLLaDA2Config,
)
from models.think_talk_llada2.modeling_think_talk_llada2 import (  # noqa: E402
    ThinkTalkLLaDA2ForCausalLM,
)


class ThinkTalkT3DInference:
    """Loads a trained T3-D checkpoint and exposes think/talk forward primitives.

    The decoding loop (generate_t3d.py) is responsible for KV-cache lifecycle
    (creation, extension, cross-block refresh); this class only provides the
    forwards.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
    ):
        self.device = device
        self.dtype = dtype

        # Resolve to absolute path; HF rejects relative paths starting with `..`
        # because they look like invalid repo IDs.
        if os.path.isdir(model_path):
            model_path = os.path.abspath(model_path)

        # Load config; VeOmni's registry uses the `_veomni` suffix to dispatch to
        # the local modeling file rather than HuggingFace's AutoModel fallback.
        config = ThinkTalkLLaDA2Config.from_pretrained(model_path)
        if not config.model_type.endswith("_veomni"):
            config.model_type = config.model_type + "_veomni"
        if getattr(config, "moe_implementation", None) != "fused":
            config.moe_implementation = "fused"
        config.use_cache = True

        self.config = config
        self.model = ThinkTalkLLaDA2ForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
        # Gradient checkpointing forces use_cache=False inside LLaDA2MoeModel
        # (line ~1035). Inference must have it disabled.
        if hasattr(self.model.model, "gradient_checkpointing"):
            self.model.model.gradient_checkpointing = False

        self.model.eval().to(device)

        # HF's AutoTokenizer.from_pretrained interprets non-absolute strings as
        # HuggingFace repo IDs. Resolve to absolute path when the arg points at
        # a local directory so `../LLaDA2.0-mini-moe-merge` works.
        tok_path = tokenizer_path or model_path
        if os.path.isdir(tok_path):
            tok_path = os.path.abspath(tok_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

        # Resolved knobs.
        self.mask_id = getattr(config, "mask_token_id", 156895)
        self.eos_id = getattr(self.tokenizer, "eos_token_id", 156892) or 156892
        self.pad_id = getattr(self.tokenizer, "pad_token_id", 156892) or 156892

    # ------------------------------------------------------------------ think
    @torch.no_grad()
    def prefill_think(
        self, input_ids: torch.LongTensor
    ) -> Tuple[torch.Tensor, object]:
        """Run think on the prompt (single-L, no doubling). Returns
        (anchor_prompt [B, prompt_len, D], past_key_values)."""
        bsz, prompt_len = input_ids.shape[:2]
        pos = torch.arange(prompt_len, device=self.device, dtype=torch.long)
        pos = pos.unsqueeze(0).expand(bsz, -1)
        anchor, pkv = self.model.run_think_and_anchor(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=pos,
            past_key_values=None,
            use_cache=True,
        )
        return anchor, pkv

    @torch.no_grad()
    def extend_think(
        self,
        new_segment_ids: torch.LongTensor,
        past_key_values,
        start_pos: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, object]:
        """Extend think's KV cache by running on the new segment. Returns
        (anchor_for_new_segment, past_key_values').

        Args:
          new_segment_ids:  [B, seg_len] tokens to be appended.
          past_key_values:  think's cache covering [0, start_pos).
          start_pos:        absolute position at which `new_segment_ids` begins.
          attention_mask:   optional 4D additive mask [B, 1, seg_len, past_len+seg_len].
                            None -> SDPA full attention (correct for single-block extend
                            since all keys are legal block-causal targets). For the
                            cross-block refresh case (seg_len = 2*block_length), the
                            caller MUST pass a 4D mask that blocks prev-block queries
                            from seeing curr-block keys (see build_cross_block_mask in
                            generate_t3d.py and DMax's _get_cross_block_attn_mask).

        Caller is responsible for any cross-block refresh logic before calling
        this (i.e. invalidating cached KV for committed-but-stale blocks).
        """
        bsz, seg_len = new_segment_ids.shape[:2]
        pos = torch.arange(
            start_pos, start_pos + seg_len, device=self.device, dtype=torch.long,
        ).unsqueeze(0).expand(bsz, -1)

        anchor_new, pkv = self.model.run_think_and_anchor(
            input_ids=new_segment_ids,
            attention_mask=attention_mask,
            position_ids=pos,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return anchor_new, pkv

    # ------------------------------------------------------------------- talk
    @torch.no_grad()
    def forward_talk(
        self,
        block_input_ids: torch.LongTensor,
        anchor_so_far: torch.Tensor,
        block_start: int,
        block_end: int,
    ) -> torch.Tensor:
        """Run talk on the current block with the cached anchor. Returns logits
        of shape [B, block_length, vocab]."""
        return self.model.run_talk_block(
            block_input_ids=block_input_ids,
            anchor_so_far=anchor_so_far,
            block_start=block_start,
            block_end=block_end,
        )

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def trim_cache(past_key_values, keep_length: int):
        """Truncate every layer's K/V to [0, keep_length) on the seq dim.

        Used for cross-block refresh: after running think on [prev_block, curr_block]
        we want to keep the refreshed K/V for `prev_block` (which now reflects the
        committed-prior-block tokens) and discard the K/V for `curr_block` (which
        was still masked, so it's stale once curr_block starts getting revealed).
        """
        from transformers.cache_utils import DynamicCache

        if past_key_values is None:
            return None
        if isinstance(past_key_values, DynamicCache):
            new_cache = DynamicCache()
            for layer_idx in range(len(past_key_values.key_cache)):
                k = past_key_values.key_cache[layer_idx][:, :, :keep_length, :]
                v = past_key_values.value_cache[layer_idx][:, :, :keep_length, :]
                new_cache.update(k, v, layer_idx)
            return new_cache
        # Legacy tuple-of-tuples format.
        return tuple(
            (k[:, :, :keep_length, :], v[:, :, :keep_length, :])
            for (k, v) in past_key_values
        )
