# Copyright 2026 University of Sydney
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Derived in part from `dFactory/models/llada2_moe/configuration_llada2_moe.py`
# in DMax (https://github.com/czg1225/DMax), Apache-2.0.

"""Configuration for the Think-Then-Talk LLaDA-2.0 model used by T3-D milestone 1.

The think backbone is a vanilla `LLaDA2MoeModel` (DMax's existing class). The talk model is
a small dense transformer with the same hidden size as think, conditioned on a per-block
anchor (last-layer hidden state by default). This config exposes the talk-specific knobs
and inherits everything else from `LLaDA2MoeConfig`.
"""

from typing import Optional

from models.llada2_moe.configuration_llada2_moe import LLaDA2MoeConfig


class ThinkTalkLLaDA2Config(LLaDA2MoeConfig):
    """Config for `ThinkTalkLLaDA2ForCausalLM`.

    Inherits the LLaDA-2.0-mini backbone fields (hidden_size, num_hidden_layers, num_experts,
    rms_norm_eps, rope_theta, ...) and adds the Think-Then-Talk extension fields below.

    The think model uses every LLaDA2Moe* field unchanged. The talk model uses the
    `talk_*` fields below; fields set to -1 are auto-resolved to match think.
    """

    # DMax/VeOmni convention: the models/<module>/ registered config uses the `_veomni` suffix
    # so it doesn't clash with the standalone HuggingFace-loadable config sitting in
    # configs/model_configs/<name>/. The latter uses `"think_talk_llada2"` (no suffix).
    model_type = "think_talk_llada2_veomni"

    def __init__(
        self,
        # === Talk model architecture ===
        talk_num_layers: int = 2,
        talk_hidden_size: int = -1,          # -1 -> match think hidden_size
        talk_num_attention_heads: int = -1,  # -1 -> match think num_attention_heads
        talk_num_key_value_heads: int = -1,  # -1 -> match think num_key_value_heads
        talk_intermediate_size: int = -1,    # -1 -> match think intermediate_size (dense MLP)
        # === Anchor fuser (brief sec 6.3) ===
        anchor_fuser_type: str = "last_only",  # last_only | last_mid | concat_linear | gated | cross_attention
        anchor_layers: str = "last",           # "last" | comma-separated indices for non-last_only types
        # === Conditioning mechanism (brief sec 6.4) ===
        anchor_conditioning: str = "gated_residual",  # gated_residual | cross_attention | prefix_token
        anchor_injection_mode: str = "gated_residual",  # gated_residual | concat_segment
                                                #   "gated_residual": anchor added per-position into
                                                #     talk's residual stream via sigmoid(alpha) or
                                                #     fixed_gate. Talk sequence stays 2L.
                                                #   "concat_segment" (tata-style): anchor is inserted
                                                #     as a SEPARATE stream of L tokens in talk's
                                                #     sequence, distinguished by a segment embedding.
                                                #     Talk sequence becomes 3L = [noisy, anchor, clean].
                                                #     Talk's attention dynamically routes to anchor/
                                                #     noisy/clean tokens. No gated_residual modules
                                                #     are instantiated in this mode.
        anchor_inject_layers: str = "first",   # only consulted when anchor_injection_mode=gated_residual
                                                # "first" -> anchor injected only at talk layer 0
                                                # "all"   -> every talk layer gets its own
                                                #            GatedResidualConditioning module
        anchor_gate_learnable: bool = True,    # True: gate=sigmoid(alpha), alpha is nn.Parameter
                                                #   initialised from anchor_gate_init below
                                                # False: gate is fixed at anchor_gate_value, no
                                                #   sigmoid, no learnable alpha. Use for diagnosing
                                                #   training dynamics without the gate as a variable.
        anchor_gate_init: float = -2.0,         # sigmoid(-2.0) ≈ 0.12; only used when learnable
        anchor_gate_value: float = 0.2,         # fixed gate value; only used when NOT learnable
        # === Think-side ablations ===
        prune_think_last_n_layer: int = 0,     # >0 enables ablation A1.5 (warm-start talk)
        # === Training flags surfaced into config so checkpoints round-trip ===
        train_think: bool = True,
        train_talk: bool = True,
        train_lm_head: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.talk_num_layers = int(talk_num_layers)
        self.talk_hidden_size = int(talk_hidden_size)
        self.talk_num_attention_heads = int(talk_num_attention_heads)
        self.talk_num_key_value_heads = int(talk_num_key_value_heads)
        self.talk_intermediate_size = int(talk_intermediate_size)

        self.anchor_fuser_type = str(anchor_fuser_type)
        self.anchor_layers = str(anchor_layers)
        self.anchor_conditioning = str(anchor_conditioning)
        self.anchor_injection_mode = str(anchor_injection_mode)
        self.anchor_inject_layers = str(anchor_inject_layers)
        self.anchor_gate_learnable = bool(anchor_gate_learnable)
        self.anchor_gate_init = float(anchor_gate_init)
        self.anchor_gate_value = float(anchor_gate_value)

        self.prune_think_last_n_layer = int(prune_think_last_n_layer)
        self.train_think = bool(train_think)
        self.train_talk = bool(train_talk)
        self.train_lm_head = bool(train_lm_head)

        self._validate()

    # ------------------------------------------------------------------ helpers

    @property
    def resolved_talk_hidden_size(self) -> int:
        return self.hidden_size if self.talk_hidden_size == -1 else self.talk_hidden_size

    @property
    def resolved_talk_num_attention_heads(self) -> int:
        return (
            self.num_attention_heads
            if self.talk_num_attention_heads == -1
            else self.talk_num_attention_heads
        )

    @property
    def resolved_talk_num_key_value_heads(self) -> int:
        return (
            self.num_key_value_heads
            if self.talk_num_key_value_heads == -1
            else self.talk_num_key_value_heads
        )

    @property
    def resolved_talk_intermediate_size(self) -> int:
        return (
            self.intermediate_size
            if self.talk_intermediate_size == -1
            else self.talk_intermediate_size
        )

    @property
    def think_num_layers_after_prune(self) -> int:
        return self.num_hidden_layers - self.prune_think_last_n_layer

    def resolved_anchor_layer_indices(self) -> list:
        """Resolve `anchor_layers` against the pruned think depth.

        - "last" -> [think_num_layers_after_prune - 1]
        - comma-separated ints -> validated; must be in range [0, think_num_layers_after_prune-1]
        """
        depth = self.think_num_layers_after_prune
        s = self.anchor_layers.strip().lower()
        if s == "last":
            return [depth - 1]
        if s in ("last_mid", "mid_last"):
            return [depth // 2, depth - 1]
        try:
            indices = [int(x.strip()) for x in self.anchor_layers.split(",")]
        except ValueError as exc:  # noqa: TRY003
            raise ValueError(
                f"anchor_layers must be 'last', 'last_mid', or comma-separated ints; "
                f"got {self.anchor_layers!r}"
            ) from exc
        for i in indices:
            if not (0 <= i < depth):
                raise ValueError(
                    f"anchor_layers index {i} out of range for think depth {depth}"
                )
        return indices

    # ------------------------------------------------------------------ validation

    def _validate(self) -> None:
        if self.talk_num_layers < 1:
            raise ValueError(f"talk_num_layers must be >= 1, got {self.talk_num_layers}")

        if self.prune_think_last_n_layer < 0 or self.prune_think_last_n_layer >= self.num_hidden_layers:
            raise ValueError(
                f"prune_think_last_n_layer must be in [0, num_hidden_layers), "
                f"got {self.prune_think_last_n_layer} (num_hidden_layers={self.num_hidden_layers})"
            )

        if self.anchor_fuser_type not in (
            "last_only", "last_mid", "concat_linear", "gated", "cross_attention",
        ):
            raise ValueError(f"unknown anchor_fuser_type: {self.anchor_fuser_type}")

        if self.anchor_conditioning not in ("gated_residual", "cross_attention", "prefix_token"):
            raise ValueError(f"unknown anchor_conditioning: {self.anchor_conditioning}")

        if self.anchor_inject_layers not in ("first", "all"):
            raise ValueError(f"unknown anchor_inject_layers: {self.anchor_inject_layers}")

        if self.anchor_injection_mode not in ("gated_residual", "concat_segment"):
            raise ValueError(f"unknown anchor_injection_mode: {self.anchor_injection_mode}")

        # In milestone 1 we require talk to match think hidden size (no projector).
        if self.talk_hidden_size != -1 and self.talk_hidden_size != self.hidden_size:
            raise ValueError(
                "talk_hidden_size != think hidden_size is reserved for ablation A5. "
                "Milestone 1 requires talk_hidden_size = -1 (or = hidden_size)."
            )
