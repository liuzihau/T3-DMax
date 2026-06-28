# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
"""Off-cluster smoke test for the DBet training step (no VeOmni, no real weights, CPU-ok).

Builds a TINY DbetForDraftDecoding (small all-dense heavy), fabricates a dual-stream batch exactly like
`train_dbet.py` builds it (left-to-right reveal + block-diffusion mask), and runs `dbet_train_step` for a few
steps. Verifies the wiring end-to-end: heavy frozen, drafter trains, loss finite and moving.

    python smoke_dbet.py            # run from .../dFactory/tasks  (needs torch; CPU is fine)
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import torch
from torch.optim import AdamW

_HERE = os.path.dirname(os.path.abspath(__file__))                 # .../dFactory/tasks
_DFACTORY = os.path.abspath(os.path.join(_HERE, ".."))            # .../dFactory
for _p in (_DFACTORY,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.dbet import DbetConfig, DbetForDraftDecoding          # noqa: E402
from dataset.data_transform_dbet import block_left_to_right_reveal, build_block_diffusion_attn_mask  # noqa: E402
from dbet_train_core import dbet_train_step                       # noqa: E402


def build_tiny_model(device, dtype):
    # Small ids so the heavy's embedding fits (padding_idx = pad_token_id must be < vocab_size).
    cfg = DbetConfig(
        vocab_size=64, hidden_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=128, max_position_embeddings=64,
        first_k_dense_replace=8,            # >= num_hidden_layers -> all-dense heavy (no MoE)
        draft_num_layers=2, sel_layers="1,2,3", warmstart_from_heavy_bottom=False,
        use_confidence_head=True,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, mask_token_id=63,   # all within vocab=64
    )
    model = DbetForDraftDecoding(cfg).to(device=device, dtype=dtype)
    model._apply_freeze_flags()
    model.train()
    return model, cfg


def make_batch(B, L, block_size, prompt_len, vocab, device, mask_id):
    clean = torch.randint(0, mask_id, (B, L))                  # 0..mask_id-1 so clean tokens != mask_id
    maskable = torch.arange(L) >= prompt_len
    noisy = torch.stack([
        block_left_to_right_reveal(clean[b], (0.5, 0.5), maskable, mask_id, block_size)
        for b in range(B)
    ])
    full = torch.cat([noisy, clean], dim=1).to(device)            # [B, 2L] = [noisy | clean]
    pos = torch.cat([torch.arange(L), torch.arange(L)]).unsqueeze(0).expand(B, -1).to(device)
    attn = build_block_diffusion_attn_mask(L, block_size, torch.float32, device).expand(B, -1, -1, -1)
    return {
        "input_ids": full, "noisy_input_ids": noisy.to(device),
        "attention_mask": attn, "position_ids": pos,
    }


def main():
    torch.manual_seed(0)
    device, dtype = "cpu", torch.float32
    B, L, block_size, prompt_len, vocab = 2, 16, 8, 4, 64

    model, cfg = build_tiny_model(device, dtype)
    args = SimpleNamespace(train=SimpleNamespace(block_size=block_size, heavy_commit_threshold=0.9, conf_loss_weight=1.0))

    # freeze checks
    n_heavy_grad = sum(p.requires_grad for p in model.heavy.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_heavy_grad == 0, f"heavy must be frozen, but {n_heavy_grad} params require grad"
    assert n_train > 0, "drafter has no trainable params"
    print(f"[smoke] trainable={n_train:,}  heavy frozen OK")

    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    mask_id = cfg.mask_token_id
    batch = make_batch(B, L, block_size, prompt_len, vocab, device, mask_id)

    first = None
    for step in range(30):
        loss, m = dbet_train_step(model, batch, n_micro_batches=1, args=args, mask_id=mask_id, return_metrics=True)
        opt.zero_grad(); loss.backward(); opt.step()
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        if step == 0:
            first = float(loss)
            # a drafter param must receive grad
            g = [p.grad for p in model.draft.parameters() if p.grad is not None]
            assert len(g) > 0, "no gradient reached the drafter"
        if step % 5 == 0 or step == 29:
            print(f"[{step:3d}] loss {float(loss):.4f}  " + "  ".join(f"{k} {v:.3f}" for k, v in m.items()))

    print(f"[smoke] loss {first:.4f} -> {float(loss):.4f}  ({'DOWN' if float(loss) < first else 'no-decrease'})")
    print("[smoke] DBet train step wiring OK")


if __name__ == "__main__":
    main()
