# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
"""Config for the **DBet** drafter — the "Self-Conditioned Δh Drafter".

`DbetConfig` extends `LLaDA2MoeConfig`, so it inherits the *heavy* (DMax / LLaDA-2.0-MoE) backbone
hyper-params verbatim (hidden_size, num_attention_heads, head_dim, rope_theta, rms_norm_eps, vocab_size, …)
— these describe the frozen heavy whose embedding / lm_head / hidden the drafter reuses — and adds the
drafter-specific fields. Sizes set to `-1` resolve to the matching heavy value (the `think_talk_llada2`
convention), so the default drafter mirrors the heavy's widths (which is also what warm-start needs).
"""

from __future__ import annotations

from ..llada2_moe.configuration_llada2_moe import LLaDA2MoeConfig


class DbetConfig(LLaDA2MoeConfig):
    """Configuration for `DbetForDraftDecoding`. See module docstring for the inheritance rationale."""

    # ends in "_veomni" so the heavy's LLaDA2MoeSparseMoeBlock uses the FUSED experts layout that matches
    # DMax's merged-MoE checkpoints (LLaDA2MoeSparseMoeBlock keys off model_type.endswith("_veomni")).
    model_type = "dbet_veomni"

    def __init__(
        self,
        # === Drafter body architecture (a thin LLaDA-2.0-shaped stack) ===
        draft_num_layers: int = 5,             # L: number of draft layers
        draft_hidden_size: int = -1,           # -1 -> match heavy hidden_size
        draft_num_attention_heads: int = -1,   # -1 -> match heavy num_attention_heads
        draft_num_key_value_heads: int = -1,   # -1 -> match heavy num_key_value_heads (GQA)
        draft_intermediate_size: int = -1,     # -1 -> match heavy intermediate_size
        draft_layer_type: str = "dense",       # body layer: "dense" (SwiGLU) | "moe" (LLaDA2MoeDecoderLayer) [moe=TODO]
        draft_hidden_act: str = "silu",        # activation for every DbetGatedMLP (ACT2FN key; silu -> liger fast path)
        position_embedding_type: str = "rope",  # position scheme; follows the heavy ("rope" = rotary; reuses rope_theta etc.)
        # === Conditioning: which heavy layers feed the fuses, and the fuse topology ===
        sel_layers: str = "1,10,19",           # comma-sep heavy layer indices read by the fuses (m = count); shallow->deep
        per_layer_prefix_fuse: bool = True,    # prefix fuse: True = 1 shared trunk -> L per-layer heads (~Lx cheaper
                                                #   than L independent fuses); False = 1 shared feature (DFlash)
        # === Gated-MLP widths (every learned projection is a DbetGatedMLP) ===
        fuse_hidden_size: int = -1,            # intermediate width of the fuse + soft-embed MLPs; -1 -> draft_hidden_size
        head_intermediate_size: int = -1,      # intermediate width of the Δh + confidence head MLPs; -1 -> draft_intermediate_size
        # === Self-conditioning soft-embed (DiffusionGemma §2.1) ===
        soft_embed_temp: float = 0.8,          # τ at round start
        soft_embed_temp_min: float = 0.4,      # τ anneal target across rounds (0.8 -> 0.4); == temp disables anneal
        # === Heads ===
        use_delta_head: bool = True,           # Δh residual on heavy's last hidden -> frozen lm_head (zero-init; step0 == heavy)
        use_confidence_head: bool = True,      # trained "will the heavy accept this draft?" head; off -> no abstention
        # === Frozen-from-heavy + warm-start ===
        freeze_embedding: bool = True,         # heavy W_E reused, frozen
        freeze_lm_head: bool = True,           # heavy lm_head reused, frozen
        freeze_final_norm: bool = True,        # heavy final norm reused, frozen
        train_heavy: bool = False,             # heavy = DMax, ALWAYS frozen; flag exists only to round-trip
        warmstart_from_heavy_bottom: bool = True,  # init the L draft layers from the heavy's bottom L decoder layers
        heavy_path: str | None = None,         # checkpoint dir of the frozen heavy (DMax-Math-16B), for loading + warm-start
        mask_token_id: int = 156895,           # LLaDA2/DMax [MASK] id; used at inference to split committed/canvas
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mask_token_id = int(mask_token_id)
        self.draft_num_layers = int(draft_num_layers)
        self.draft_hidden_size = int(draft_hidden_size)
        self.draft_num_attention_heads = int(draft_num_attention_heads)
        self.draft_num_key_value_heads = int(draft_num_key_value_heads)
        self.draft_intermediate_size = int(draft_intermediate_size)
        self.draft_layer_type = str(draft_layer_type)
        self.draft_hidden_act = str(draft_hidden_act)
        self.position_embedding_type = str(position_embedding_type)
        self.sel_layers = sel_layers
        self.per_layer_prefix_fuse = bool(per_layer_prefix_fuse)
        self.fuse_hidden_size = int(fuse_hidden_size)
        self.head_intermediate_size = int(head_intermediate_size)
        self.soft_embed_temp = float(soft_embed_temp)
        self.soft_embed_temp_min = float(soft_embed_temp_min)
        self.use_delta_head = bool(use_delta_head)
        self.use_confidence_head = bool(use_confidence_head)
        self.freeze_embedding = bool(freeze_embedding)
        self.freeze_lm_head = bool(freeze_lm_head)
        self.freeze_final_norm = bool(freeze_final_norm)
        self.train_heavy = bool(train_heavy)
        self.warmstart_from_heavy_bottom = bool(warmstart_from_heavy_bottom)
        self.heavy_path = heavy_path

    # ---- resolvers (-1 placeholders -> concrete heavy-matched sizes) ----
    @property
    def resolved_draft_hidden_size(self) -> int:
        return self.draft_hidden_size if self.draft_hidden_size != -1 else self.hidden_size

    @property
    def resolved_draft_num_attention_heads(self) -> int:
        return self.draft_num_attention_heads if self.draft_num_attention_heads != -1 else self.num_attention_heads

    @property
    def resolved_draft_num_key_value_heads(self) -> int:
        return self.draft_num_key_value_heads if self.draft_num_key_value_heads != -1 else self.num_key_value_heads

    @property
    def resolved_draft_intermediate_size(self) -> int:
        return self.draft_intermediate_size if self.draft_intermediate_size != -1 else self.intermediate_size

    @property
    def resolved_fuse_hidden_size(self) -> int:
        return self.fuse_hidden_size if self.fuse_hidden_size != -1 else self.resolved_draft_hidden_size

    @property
    def resolved_head_intermediate_size(self) -> int:
        return self.head_intermediate_size if self.head_intermediate_size != -1 else self.resolved_draft_intermediate_size

    @property
    def draft_head_dim(self) -> int:
        # Match the heavy's head_dim by default so rotary + warm-start line up.
        return self.head_dim or (self.resolved_draft_hidden_size // self.resolved_draft_num_attention_heads)

    @property
    def sel_layers_list(self) -> list[int]:
        return sorted(int(x) for x in str(self.sel_layers).split(",") if str(x).strip() != "")

    @property
    def m(self) -> int:
        return len(self.sel_layers_list)
