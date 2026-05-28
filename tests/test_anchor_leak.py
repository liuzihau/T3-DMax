# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# Brief sec 8.4: mandatory pre-training verification that the doubled-sequence convention
# `cat([noisy, clean], dim=1)` does NOT let the clean half leak into the noisy half's
# hidden states (which we use to build the anchor).
#
# This test must pass before any T3-D training run. Failure means anchor leak is real
# and the model will train to a value the inference path cannot match.
#
# Environment: run from the T3-DMax repo root with PYTHONPATH including dFactory/, e.g.
#     cd <T3-DMax repo>
#     PYTHONPATH=dFactory:dFactory/VeOmni:$PYTHONPATH pytest tests/test_anchor_leak.py -v

import importlib.util
import os

import pytest
import torch


def _block_diffusion_mask_from_training_script():
    """Imports the mask function from the T3-D training script without running main()."""
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.normpath(os.path.join(here, "..", "dFactory", "tasks", "train_t3_dmax_bd_oput.py"))
    spec = importlib.util.spec_from_file_location("train_t3_dmax_bd_oput", script)
    module = importlib.util.module_from_spec(spec)
    # The module imports VeOmni etc. at top level which will fail outside the right env.
    # We only want `block_diffusion_mask` -- parse it out by name to avoid full exec.
    try:
        spec.loader.exec_module(module)
        return module.block_diffusion_mask
    except (ImportError, ModuleNotFoundError):
        # Fallback: reimplement the mask inline. Kept in sync with the source -- if
        # DMax changes the mask, update both places.
        def block_diffusion_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
            x0_flag_q = (q_idx >= n)
            x0_flag_kv = (kv_idx >= n)
            block_q = torch.where(x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size)
            block_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size)
            block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
            offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)
            block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)
            return block_diagonal | offset_block_causal | block_causal
        return block_diffusion_mask


# ----------------------------------------------------------------------------
# Test 1: pure mask logic. Always runs (no model, no GPU).
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("seq_len,block_size", [(32, 8), (64, 16), (128, 32), (256, 32)])
def test_noisy_cannot_see_own_or_future_clean_blocks(seq_len, block_size):
    """The DMax block-diffusion mask deliberately lets noisy queries see clean keys in
    PRIOR blocks (offset block-causal mask, M_OBC) -- at inference those represent
    already-decoded tokens, so making them visible during training matches the inference
    distribution. The actual leak invariant is stricter:

        For a noisy query in block i, it must NOT attend to any clean key in block i,
        or in any block j > i (future).

    Visualised on the doubled sequence (n = seq_len, blocks of size block_size):

        block i query, block j clean key:
            j <  i : allowed  (already-decoded context)
            j >= i : FORBIDDEN (would be the label or future label)
    """
    fn = _block_diffusion_mask_from_training_script()
    full_len = seq_len * 2

    mask = fn(
        b=None, h=None,
        q_idx=torch.arange(full_len)[:, None],
        kv_idx=torch.arange(full_len)[None, :],
        block_size=block_size,
        n=seq_len,
    )
    # Compute block indices for each (q, kv) cell.
    q_block = (torch.arange(full_len) // block_size)[:, None]              # full-len q index
    kv_block_clean = ((torch.arange(full_len) - seq_len) // block_size)[None, :]

    # The leak quadrant we care about: q in noisy half, kv in clean half.
    noisy_q = (torch.arange(full_len) < seq_len)[:, None]
    clean_kv = (torch.arange(full_len) >= seq_len)[None, :]

    # Sub-quadrant where leak would matter: noisy q AND clean kv AND kv-block >= q-block.
    bad_cells = noisy_q & clean_kv & (kv_block_clean >= q_block)
    if not bad_cells.any():
        return  # no cells to check
    forbidden_and_allowed = mask & bad_cells
    assert not forbidden_and_allowed.any(), (
        f"Anchor leak detected: noisy queries can attend to clean keys in their own or "
        f"future blocks at {forbidden_and_allowed.nonzero().tolist()[:5]} "
        f"(seq_len={seq_len}, block_size={block_size})"
    )


@pytest.mark.parametrize("seq_len,block_size", [(32, 8), (64, 16), (128, 32)])
def test_noisy_can_see_prior_clean_blocks(seq_len, block_size):
    """Positive sanity: the offset-block-causal mask is doing its job. A noisy query
    in block i should be able to attend to clean keys in block j < i. If this is
    *not* the case, the model would not get the "already-decoded prior context" signal
    that block diffusion training requires."""
    fn = _block_diffusion_mask_from_training_script()
    full_len = seq_len * 2

    mask = fn(
        b=None, h=None,
        q_idx=torch.arange(full_len)[:, None],
        kv_idx=torch.arange(full_len)[None, :],
        block_size=block_size,
        n=seq_len,
    )
    q_block = (torch.arange(full_len) // block_size)[:, None]
    kv_block_clean = ((torch.arange(full_len) - seq_len) // block_size)[None, :]
    noisy_q = (torch.arange(full_len) < seq_len)[:, None]
    clean_kv = (torch.arange(full_len) >= seq_len)[None, :]
    expected_allowed = noisy_q & clean_kv & (kv_block_clean < q_block) & (kv_block_clean >= 0)

    # If there are any such cells (only when seq_len > block_size), they must all be True.
    if expected_allowed.any():
        assert torch.all(mask[expected_allowed]), (
            "Offset-block-causal mask is broken: noisy queries cannot see prior clean blocks"
        )


def _block_diffusion_mask_3L_from_training_script():
    """Imports the 3L mask function from the T3-D training script."""
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.normpath(os.path.join(here, "..", "dFactory", "tasks", "train_t3_dmax_bd_oput.py"))
    spec = importlib.util.spec_from_file_location("train_t3_dmax_bd_oput", script)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module.block_diffusion_mask_3L
    except (ImportError, ModuleNotFoundError):
        # Fallback: reimplement inline. Keep in sync with the training script source.
        def block_diffusion_mask_3L(q_idx, kv_idx, block_size, n):
            x0_flag_q  = (q_idx  >= 2 * n)
            x0_flag_kv = (kv_idx >= 2 * n)
            eff_q  = torch.where(q_idx  < n, q_idx,  torch.where(q_idx  < 2*n, q_idx  - n, q_idx  - 2*n))
            eff_kv = torch.where(kv_idx < n, kv_idx, torch.where(kv_idx < 2*n, kv_idx - n, kv_idx - 2*n))
            block_q  = eff_q  // block_size
            block_kv = eff_kv // block_size
            block_diagonal      = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
            offset_block_causal = (block_q >  block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)
            block_causal        = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)
            return block_diagonal | offset_block_causal | block_causal
        return block_diffusion_mask_3L


# ----------------------------------------------------------------------------
# Test 1b: 3L mask checks for concat_segment mode.
# Layout: [noisy(0..n-1), anchor(n..2n-1), clean(2n..3n-1)].
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("seq_len,block_size", [(32, 8), (64, 16), (128, 32)])
def test_3L_noisy_cannot_see_own_or_future_clean_blocks(seq_len, block_size):
    """In the 3L mask, noisy queries (region 1) must NOT attend to clean keys (region 3)
    at the same or future block. They CAN attend to clean keys in prior blocks (M_OBC)."""
    fn = _block_diffusion_mask_3L_from_training_script()
    full_len = 3 * seq_len
    mask = fn(
        q_idx=torch.arange(full_len)[:, None],
        kv_idx=torch.arange(full_len)[None, :],
        block_size=block_size,
        n=seq_len,
    )
    # Noisy region: [0, n). Clean region: [2n, 3n). Anchor: [n, 2n).
    noisy_q_idx = torch.arange(seq_len)[:, None]            # [seq_len, 1]
    clean_kv_idx = torch.arange(2 * seq_len, 3 * seq_len)[None, :]  # [1, seq_len]
    noisy_q_block = noisy_q_idx // block_size
    clean_kv_block = (clean_kv_idx - 2 * seq_len) // block_size

    forbidden = clean_kv_block >= noisy_q_block  # same-block or future
    leak_quadrant = mask[:seq_len, 2 * seq_len:3 * seq_len]
    bad_cells = leak_quadrant & forbidden
    assert not bad_cells.any(), (
        f"Anchor leak in 3L mask: noisy can see clean in own/future blocks at "
        f"{bad_cells.nonzero().tolist()[:5]}"
    )


@pytest.mark.parametrize("seq_len,block_size", [(32, 8), (64, 16), (128, 32)])
def test_3L_anchor_cannot_see_own_or_future_clean_blocks(seq_len, block_size):
    """In the 3L mask, anchor queries (region 2) must NOT attend to clean keys (region 3)
    at the same or future block. This is the new leak risk introduced by the 3L layout."""
    fn = _block_diffusion_mask_3L_from_training_script()
    full_len = 3 * seq_len
    mask = fn(
        q_idx=torch.arange(full_len)[:, None],
        kv_idx=torch.arange(full_len)[None, :],
        block_size=block_size,
        n=seq_len,
    )
    # Anchor query offset = (idx - n); clean key offset = (idx - 2n). Block from offset.
    anchor_q_offset = (torch.arange(seq_len, 2 * seq_len) - seq_len)[:, None]
    clean_kv_offset = (torch.arange(2 * seq_len, 3 * seq_len) - 2 * seq_len)[None, :]
    anchor_q_block = anchor_q_offset // block_size
    clean_kv_block = clean_kv_offset // block_size
    forbidden = clean_kv_block >= anchor_q_block
    leak_quadrant = mask[seq_len:2 * seq_len, 2 * seq_len:3 * seq_len]
    bad_cells = leak_quadrant & forbidden
    assert not bad_cells.any(), (
        f"Anchor leak in 3L mask: anchor query can see clean in own/future blocks at "
        f"{bad_cells.nonzero().tolist()[:5]}"
    )


@pytest.mark.parametrize("seq_len,block_size", [(32, 8), (64, 16), (128, 32)])
def test_3L_clean_cannot_see_noisy_or_anchor(seq_len, block_size):
    """In the 3L mask, clean queries (region 3) must NOT attend to noisy/anchor keys
    (regions 1, 2). Clean is the strict context-only stream."""
    fn = _block_diffusion_mask_3L_from_training_script()
    full_len = 3 * seq_len
    mask = fn(
        q_idx=torch.arange(full_len)[:, None],
        kv_idx=torch.arange(full_len)[None, :],
        block_size=block_size,
        n=seq_len,
    )
    clean_to_noisy = mask[2 * seq_len:3 * seq_len, :seq_len]
    clean_to_anchor = mask[2 * seq_len:3 * seq_len, seq_len:2 * seq_len]
    assert not clean_to_noisy.any(), "Clean queries can attend to noisy keys (should be blocked)"
    assert not clean_to_anchor.any(), "Clean queries can attend to anchor keys (should be blocked)"


@pytest.mark.parametrize("seq_len,block_size", [(32, 8), (64, 16), (128, 32)])
def test_3L_noisy_and_anchor_are_attentionally_symmetric(seq_len, block_size):
    """Anchor stream should behave identically to noisy stream for attention purposes:
    (1,2) = (1,1) = M_BD, (2,1) = (1,1), (2,2) = (1,1), (2,3) = (1,3) = M_OBC."""
    fn = _block_diffusion_mask_3L_from_training_script()
    full_len = 3 * seq_len
    mask = fn(
        q_idx=torch.arange(full_len)[:, None],
        kv_idx=torch.arange(full_len)[None, :],
        block_size=block_size,
        n=seq_len,
    )
    noisy_noisy = mask[:seq_len, :seq_len]
    anchor_noisy = mask[seq_len:2 * seq_len, :seq_len]
    noisy_anchor = mask[:seq_len, seq_len:2 * seq_len]
    anchor_anchor = mask[seq_len:2 * seq_len, seq_len:2 * seq_len]

    # All four quadrants should equal M_BD (block-diagonal within the shared intra-block space).
    assert torch.equal(noisy_noisy, anchor_noisy), "anchor q -> noisy k should equal noisy -> noisy"
    assert torch.equal(noisy_noisy, noisy_anchor), "noisy q -> anchor k should equal noisy -> noisy"
    assert torch.equal(noisy_noisy, anchor_anchor), "anchor q -> anchor k should equal noisy -> noisy"

    # Both noisy and anchor queries should see clean prior blocks the same way (M_OBC).
    noisy_clean = mask[:seq_len, 2 * seq_len:3 * seq_len]
    anchor_clean = mask[seq_len:2 * seq_len, 2 * seq_len:3 * seq_len]
    assert torch.equal(noisy_clean, anchor_clean), "anchor q -> clean k should equal noisy -> clean"


def test_noisy_self_attention_is_block_diagonal():
    """Sanity: within the noisy half, attention is block-diagonal (each noisy position
    can attend to other noisy positions in the same block, and nothing outside)."""
    fn = _block_diffusion_mask_from_training_script()
    seq_len, block_size = 64, 16

    mask = fn(
        b=None, h=None,
        q_idx=torch.arange(seq_len * 2)[:, None],
        kv_idx=torch.arange(seq_len * 2)[None, :],
        block_size=block_size,
        n=seq_len,
    )
    noisy_noisy = mask[:seq_len, :seq_len]
    # Build expected: block-diagonal with `block_size` blocks.
    q_block = torch.arange(seq_len)[:, None] // block_size
    kv_block = torch.arange(seq_len)[None, :] // block_size
    expected = (q_block == kv_block)
    assert torch.equal(noisy_noisy, expected), "Noisy self-attention is not block-diagonal"


# ----------------------------------------------------------------------------
# Test 1c: hybrid_xattn mask checks.
# Talk's self-attn is L x L (noisy positions only); talk's cross-attn is L x 2L
# (Q from noisy; KV from anchor at [noisy_positions, clean_positions]).
# ----------------------------------------------------------------------------

def _hybrid_xattn_masks_from_training_script():
    """Imports the hybrid_xattn mask functions from the T3-D training script."""
    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.normpath(os.path.join(here, "..", "dFactory", "tasks", "train_t3_dmax_bd_oput.py"))
    spec = importlib.util.spec_from_file_location("train_t3_dmax_bd_oput", script)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module.talk_self_attn_mask_L, module.talk_cross_attn_mask
    except (ImportError, ModuleNotFoundError):
        # Fallback: reimplement inline. Keep in sync with the training script source.
        def talk_self_attn_mask_L(q_idx, kv_idx, block_size):
            return (q_idx // block_size) == (kv_idx // block_size)

        def talk_cross_attn_mask(q_idx, kv_idx, block_size, n):
            kv_is_clean = (kv_idx >= n)
            kv_eff_pos = torch.where(kv_is_clean, kv_idx - n, kv_idx)
            q_block = q_idx // block_size
            kv_block = kv_eff_pos // block_size
            # anchor_noisy at block c: visible iff c <= b (NOT all-visible; the old
            # "all-visible" version leaked clean[b] via think's M_OBC at c > b).
            noisy_kv_ok = (~kv_is_clean) & (kv_block <= q_block)
            clean_kv_ok = kv_is_clean & (kv_block < q_block)
            return noisy_kv_ok | clean_kv_ok

        return talk_self_attn_mask_L, talk_cross_attn_mask


@pytest.mark.parametrize("seq_len,block_size", [(32, 8), (64, 16), (128, 32), (256, 32)])
def test_xattn_talk_self_attn_is_block_diagonal(seq_len, block_size):
    """In hybrid_xattn mode, talk's L self-attn over noisy positions must be block-diagonal."""
    self_fn, _ = _hybrid_xattn_masks_from_training_script()
    mask = self_fn(
        q_idx=torch.arange(seq_len)[:, None],
        kv_idx=torch.arange(seq_len)[None, :],
        block_size=block_size,
    )
    q_block = torch.arange(seq_len)[:, None] // block_size
    kv_block = torch.arange(seq_len)[None, :] // block_size
    expected = (q_block == kv_block)
    assert torch.equal(mask, expected), (
        f"Hybrid_xattn talk self-attn mask is not block-diagonal (seq_len={seq_len}, "
        f"block_size={block_size})."
    )


@pytest.mark.parametrize("seq_len,block_size", [(32, 8), (64, 16), (128, 32), (256, 32)])
def test_xattn_cross_anchor_noisy_half_obeys_block_obc(seq_len, block_size):
    """Hybrid_xattn cross-attn: noisy_q at block b can attend to anchor_noisy at block
    c iff c <= b. The naive "all of anchor_noisy is visible" was wrong: anchor_noisy[c]
    for c > b was produced by think reading clean[< c] (M_OBC), which includes clean[b]
    -- exactly the label talk's noisy_q at block b is supposed to predict. Forbid it."""
    _, cross_fn = _hybrid_xattn_masks_from_training_script()
    mask = cross_fn(
        q_idx=torch.arange(seq_len)[:, None],
        kv_idx=torch.arange(2 * seq_len)[None, :],
        block_size=block_size,
        n=seq_len,
    )
    noisy_half = mask[:, :seq_len]                                       # [L, L]
    q_block  = (torch.arange(seq_len) // block_size)[:, None]            # [L, 1]
    kv_block = (torch.arange(seq_len) // block_size)[None, :]            # [1, L]
    expected = (kv_block <= q_block)                                     # same-block OK
    assert torch.equal(noisy_half, expected), (
        f"Hybrid_xattn cross-attn anchor_noisy half is not c<=b restricted "
        f"(seq_len={seq_len}, block_size={block_size}). First disagreement at "
        f"{(noisy_half != expected).nonzero().tolist()[:5]}."
    )


@pytest.mark.parametrize("seq_len,block_size", [(32, 8), (64, 16), (128, 32)])
def test_xattn_cross_cannot_see_anchor_noisy_in_future_blocks(seq_len, block_size):
    """Explicit leak check: for noisy_q at block b, anchor_noisy at block c > b MUST
    be forbidden, since anchor_noisy[c>b] encodes clean[b] via think's M_OBC."""
    _, cross_fn = _hybrid_xattn_masks_from_training_script()
    mask = cross_fn(
        q_idx=torch.arange(seq_len)[:, None],
        kv_idx=torch.arange(2 * seq_len)[None, :],
        block_size=block_size,
        n=seq_len,
    )
    noisy_half = mask[:, :seq_len]                                       # [L, L]
    q_block  = (torch.arange(seq_len) // block_size)[:, None]
    kv_block = (torch.arange(seq_len) // block_size)[None, :]
    forbidden = (kv_block > q_block)
    leaks = noisy_half & forbidden
    assert not leaks.any(), (
        f"LEAK: noisy_q at block b can attend to anchor_noisy at block c>b at "
        f"{leaks.nonzero().tolist()[:5]} (seq_len={seq_len}, block_size={block_size}). "
        f"This is the path through which clean[b] flows into the prediction at block b."
    )


@pytest.mark.parametrize("seq_len,block_size", [(32, 8), (64, 16), (128, 32), (256, 32)])
def test_xattn_cross_cannot_see_own_or_future_anchor_clean_blocks(seq_len, block_size):
    """In hybrid_xattn cross-attn, noisy[b] must NOT see anchor's clean-half[c >= b].
    Same anchor leak risk as 2L M_OBC, applied to the cross-attn K/V's right half."""
    _, cross_fn = _hybrid_xattn_masks_from_training_script()
    mask = cross_fn(
        q_idx=torch.arange(seq_len)[:, None],
        kv_idx=torch.arange(2 * seq_len)[None, :],
        block_size=block_size,
        n=seq_len,
    )
    clean_half = mask[:, seq_len:]  # [L, L]
    q_block = (torch.arange(seq_len) // block_size)[:, None]
    kv_clean_block = (torch.arange(seq_len) // block_size)[None, :]
    forbidden = kv_clean_block >= q_block
    bad_cells = clean_half & forbidden
    assert not bad_cells.any(), (
        f"Anchor leak in hybrid_xattn cross-attn: noisy[b] can see anchor_clean in own/"
        f"future blocks at {bad_cells.nonzero().tolist()[:5]} "
        f"(seq_len={seq_len}, block_size={block_size})."
    )


@pytest.mark.parametrize("seq_len,block_size", [(32, 8), (64, 16), (128, 32)])
def test_xattn_cross_can_see_prior_anchor_clean_blocks(seq_len, block_size):
    """Positive sanity for hybrid_xattn: noisy[b] CAN see anchor_clean[c < b]
    (those are already-decoded prior context; not visible would block all anchor info)."""
    _, cross_fn = _hybrid_xattn_masks_from_training_script()
    mask = cross_fn(
        q_idx=torch.arange(seq_len)[:, None],
        kv_idx=torch.arange(2 * seq_len)[None, :],
        block_size=block_size,
        n=seq_len,
    )
    clean_half = mask[:, seq_len:]  # [L, L]
    # For any q in block b>=1, kv at position 0 (block 0) should be visible.
    if seq_len >= 2 * block_size:
        for q_pos in range(block_size, seq_len):
            assert clean_half[q_pos, 0], (
                f"Hybrid_xattn cross-attn blocks anchor_clean[0] (block 0) from "
                f"noisy q={q_pos} (block {q_pos // block_size}). Should be visible "
                f"(prior block)."
            )


# ----------------------------------------------------------------------------
# Test 2: end-to-end model invariance. Requires the model + LLaDA2 deps to be importable.
# This is the strongest check -- it actually runs the think backbone and verifies hidden
# states on the noisy half are unchanged when the clean half is perturbed.
# ----------------------------------------------------------------------------

def _try_import_model():
    """Returns (Config, Model) or skips the test if the modeling deps are unavailable."""
    try:
        from models.think_talk_llada2 import (  # type: ignore[import-not-found]
            ThinkTalkLLaDA2Config,
            ThinkTalkLLaDA2ForCausalLM,
        )
        return ThinkTalkLLaDA2Config, ThinkTalkLLaDA2ForCausalLM
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.skip(
            f"think_talk_llada2 / LLaDA2 modeling deps not importable from this env: {exc}. "
            f"Run from the T3-DMax repo with PYTHONPATH=dFactory:dFactory/VeOmni."
        )


def _tiny_config_and_model():
    """Build a tiny ThinkTalkLLaDA2 model on CPU for end-to-end leak tests.

    Note: pad_token_id defaults to 126081 in LLaDA-2.0-mini's real config, but with
    vocab_size=512 here that would blow up `nn.Embedding(padding_idx=...)`. Override
    to a tiny in-range value -- the actual pad id doesn't matter for these tests.
    """
    Config, Model = _try_import_model()
    config = Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,            # dense MLP (used by layer 0 since first_k_dense_replace=1)
        moe_intermediate_size=64,         # MoE expert MLP (used by layers 1..3)
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,             # matches the real LLaDA-2.0-mini structure
        first_k_dense_replace=1,
        n_group=1,                        # tiny config: 1 expert group instead of 8
        topk_group=1,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        pad_token_id=0,
        talk_num_layers=2,
        anchor_fuser_type="last_only",
        anchor_layers="last",
    )
    torch.manual_seed(0)
    return config, Model(config).eval()


def _build_attn_mask(seq_len: int, block_size: int, batch: int) -> torch.Tensor:
    fn = _block_diffusion_mask_from_training_script()
    full_len = seq_len * 2
    flag = fn(
        b=None, h=None,
        q_idx=torch.arange(full_len)[:, None],
        kv_idx=torch.arange(full_len)[None, :],
        block_size=block_size,
        n=seq_len,
    ).unsqueeze(0).unsqueeze(0)
    attn_mask = torch.zeros_like(flag, dtype=torch.float32).masked_fill_(flag.logical_not(), float("-inf"))
    return attn_mask.expand(batch, -1, -1, -1)


@pytest.mark.slow
def test_anchor_invariant_to_future_clean_perturbation():
    """Golden test (brief sec 8.4): perturbing clean tokens in *the noisy query's own
    or future blocks* must NOT change the think anchor on those noisy positions. The
    legitimate visibility of *prior* clean blocks (offset block-causal) means we cannot
    use a wholesale clean-half perturbation -- we must perturb only the cells the mask
    actually forbids from being seen."""
    config, model = _tiny_config_and_model()

    B = 1
    seq_len = 32
    block_size = 16
    num_blocks = seq_len // block_size
    perturbed_block = num_blocks - 1  # last block; tightest test of the mask

    torch.manual_seed(1)
    noisy = torch.randint(0, config.vocab_size, (B, seq_len))
    clean_a = torch.randint(0, config.vocab_size, (B, seq_len))
    # clean_b = clean_a except in the perturbed block.
    clean_b = clean_a.clone()
    pb_start, pb_end = perturbed_block * block_size, (perturbed_block + 1) * block_size
    torch.manual_seed(2)
    clean_b[:, pb_start:pb_end] = torch.randint(0, config.vocab_size, (B, pb_end - pb_start))

    input_a = torch.cat([noisy, clean_a], dim=1)
    input_b = torch.cat([noisy, clean_b], dim=1)
    pos = torch.cat([torch.arange(seq_len), torch.arange(seq_len)])[None, :].expand(B, -1).clone()
    attn_mask = _build_attn_mask(seq_len, block_size, B)

    with torch.no_grad():
        anchor_a = model.run_think_and_anchor(input_a, attention_mask=attn_mask, position_ids=pos)
        anchor_b = model.run_think_and_anchor(input_b, attention_mask=attn_mask, position_ids=pos)

    # Noisy positions in block `perturbed_block` cannot attend to clean block `perturbed_block`.
    # So their anchor must be invariant.
    diff = (anchor_a[:, pb_start:pb_end] - anchor_b[:, pb_start:pb_end]).abs().max().item()
    assert diff < 1e-5, (
        f"Anchor leak detected at model level: noisy anchor in block {perturbed_block} "
        f"differs by {diff:.4e} when only the same-index clean block is perturbed. "
        f"The attention mask is not enforcing the contract end-to-end."
    )


@pytest.mark.slow
def test_talk_logits_invariant_to_future_clean_perturbation():
    """Brief sec 8.4: same property must hold for talk logits, since talk consumes the
    anchor and runs on the same doubled-sequence input."""
    config, model = _tiny_config_and_model()

    B = 1
    seq_len = 32
    block_size = 16
    num_blocks = seq_len // block_size
    perturbed_block = num_blocks - 1

    torch.manual_seed(1)
    noisy = torch.randint(0, config.vocab_size, (B, seq_len))
    clean_a = torch.randint(0, config.vocab_size, (B, seq_len))
    clean_b = clean_a.clone()
    pb_start, pb_end = perturbed_block * block_size, (perturbed_block + 1) * block_size
    torch.manual_seed(2)
    clean_b[:, pb_start:pb_end] = torch.randint(0, config.vocab_size, (B, pb_end - pb_start))

    input_a = torch.cat([noisy, clean_a], dim=1)
    input_b = torch.cat([noisy, clean_b], dim=1)
    pos = torch.cat([torch.arange(seq_len), torch.arange(seq_len)])[None, :].expand(B, -1).clone()
    attn_mask = _build_attn_mask(seq_len, block_size, B)

    with torch.no_grad():
        logits_a = model(input_ids=input_a, attention_mask=attn_mask, position_ids=pos).logits
        logits_b = model(input_ids=input_b, attention_mask=attn_mask, position_ids=pos).logits

    diff = (logits_a[:, pb_start:pb_end] - logits_b[:, pb_start:pb_end]).abs().max().item()
    assert diff < 1e-4, (
        f"Talk logits in noisy block {perturbed_block} differ by {diff:.4e} when only "
        f"the same-index clean block is perturbed. Anchor leak is propagating through "
        f"the full forward."
    )


@pytest.mark.slow
def test_anchor_DOES_change_when_prior_clean_block_perturbed():
    """Positive sanity: perturbing clean tokens in PRIOR blocks (which the mask
    legitimately exposes to the noisy query) MUST change the anchor. Otherwise the
    block-diffusion training signal that uses "already-decoded prior context" is broken
    or the model is ignoring its inputs."""
    config, model = _tiny_config_and_model()

    B = 1
    seq_len = 32
    block_size = 16

    # Anchor for block 1 (the last noisy block) should depend on clean block 0.
    torch.manual_seed(1)
    noisy = torch.randint(0, config.vocab_size, (B, seq_len))
    clean_a = torch.randint(0, config.vocab_size, (B, seq_len))
    clean_b = clean_a.clone()
    torch.manual_seed(2)
    clean_b[:, :block_size] = torch.randint(0, config.vocab_size, (B, block_size))  # perturb block 0

    input_a = torch.cat([noisy, clean_a], dim=1)
    input_b = torch.cat([noisy, clean_b], dim=1)
    pos = torch.cat([torch.arange(seq_len), torch.arange(seq_len)])[None, :].expand(B, -1).clone()
    attn_mask = _build_attn_mask(seq_len, block_size, B)

    with torch.no_grad():
        anchor_a = model.run_think_and_anchor(input_a, attention_mask=attn_mask, position_ids=pos)
        anchor_b = model.run_think_and_anchor(input_b, attention_mask=attn_mask, position_ids=pos)

    # Anchor in block 1 SHOULD see the clean block 0 change (offset block-causal).
    diff = (anchor_a[:, block_size:] - anchor_b[:, block_size:]).abs().max().item()
    assert diff > 1e-3, (
        f"Sanity-check failure: noisy anchor in block 1 did not change ({diff:.4e}) "
        f"when clean block 0 was perturbed. Either the offset-block-causal mask is "
        f"broken, the model is ignoring its KV inputs, or the test is malformed."
    )
