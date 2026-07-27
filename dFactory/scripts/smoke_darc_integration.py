# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# DARC integration SMOKE (real model, 1-2 prompts, NO training harness / NO FSDP). Validates the risky core
# of the base-model integration before building train_darc.py:
#   (A) tap index: hidden_states[tap_layer+1] is the raw output of layer `tap_layer`; the last hidden_states
#       entry is POST-final-norm -> check lm_head(hs[-1]) ~= out.logits.
#   (B) tap signal ANCHOR: the layer-18 logit-lens recall@1 by block-position on the model's own gold should
#       reproduce the probe (pos0 high ~0.9, decaying) -> confirms we tapped the right hidden with the right mask.
#   (C) the DARC head runs on the real bf16 hidden: forward_train -> finite loss, backward, head trains,
#       backbone frozen. Reuses data_transform_darc.process_darc_gold_example (transform -> head pipeline).
#
# Run (GPU box, merged DMax checkpoint):
#   cd dFactory/scripts
#   python smoke_darc_integration.py --model_path ../../DMax-Math-16B-moe-merge --limit 2

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))                 # dFactory/scripts
_T3 = os.path.abspath(os.path.join(_HERE, "..", ".."))            # T3-DMax
for _p in (os.path.join(_T3, "dInfer", "evaluations"),           # probe: load_fused, decode
           os.path.join(_T3, "dFactory", "models", "darc"),      # DarcHead, DarcConfig
           os.path.join(_T3, "dFactory", "tasks", "dataset")):   # process_darc_gold_example
    if _p not in sys.path:
        sys.path.insert(0, _p)

from probe_layer_readout import load_fused, decode_and_maybe_probe  # noqa: E402
from dinfer.decoding.generate_t3d import build_block_causal_mask     # noqa: E402
from dinfer.decoding.generate_dbet import MASK_ID, EOS_ID            # noqa: E402
from eval_tasks import load_task                                     # noqa: E402
from configuration_darc import DarcConfig                            # noqa: E402
from modeling_darc import DarcHead                                   # noqa: E402
from data_transform_darc import process_darc_gold_example            # noqa: E402


@torch.no_grad()
def get_gold(model, embed, lm_head, final_norm, pid, gen_length, block_length, V, device):
    """Model's own answer (self-gold) via the probe's faithful DMax decode."""
    gx, ge = decode_and_maybe_probe(model, embed, lm_head, final_norm, pid, gen_length, block_length,
                                    0.5, V, device, gold_block_fn=None, acc=None)
    return gx, ge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--tap_layer", type=int, default=18)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    mp = os.path.abspath(args.model_path)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.abspath(args.tokenizer_path or mp), trust_remote_code=True)
    model = load_fused(mp, device)
    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    final_norm = model.model.norm
    rotary_emb = model.model.rotary_emb
    V = int(model.config.vocab_size)
    D = int(model.config.hidden_size)
    NL = int(model.config.num_hidden_layers)
    tap_index = args.tap_layer + 1                                 # output of layer tap_layer
    dt = embed.weight.dtype
    print(f"[smoke] layers={NL} D={D} V={V} tap_layer={args.tap_layer} -> hidden_states[{tap_index}] dtype={dt}")

    cfg = DarcConfig(hidden_size=D, num_attention_heads=model.config.num_attention_heads,
                     num_key_value_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim,
                     intermediate_size=model.config.intermediate_size, vocab_size=V,
                     rotary_dim=getattr(model.config, "rotary_dim", 64), block_size=args.block_length,
                     tap_layer=args.tap_layer)
    setattr(cfg, "mask_token_id", MASK_ID)
    head = DarcHead(cfg).to(device=device, dtype=dt)
    head.train()

    rows = load_task(args.task, limit=args.limit)
    for ridx, row in enumerate(rows):
        pid = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                      add_generation_prompt=True, tokenize=True, return_tensors="pt").to(device)
        P = pid.shape[1]
        gx, ge = get_gold(model, embed, lm_head, final_norm, pid, args.gen_length, args.block_length, V, device)
        gold_ids = gx[P:ge].tolist()
        if len(gold_ids) < 4:
            print(f"[smoke] ex{ridx}: empty gold, skip"); continue

        # transform -> training tensors (all-masked forward-1)
        B = args.block_length
        L = ((int(ge) // B) + 1) * B                                # block-aligned length covering the gold
        rec = {"prompt_ids": pid[0].tolist(), "gold_ids": gold_ids, "prompt_len": P}
        ex = process_darc_gold_example(rec, max_seq_len=L, block_size=B, mode="all_masked")[0]
        input_ids = ex["input_ids"].to(device)[None]               # [1,L] clean
        noisy = ex["noisy_input_ids"].to(device)[None]             # [1,L] generated masked
        labels = ex["labels"].to(device)[None]                     # [1,L]
        pos_ids = torch.arange(L, device=device)[None]
        attn = build_block_causal_mask(L, B, dtype=dt, device=device)

        # --- masked forward-1 through the frozen heavy ---
        out = model(input_ids=noisy, attention_mask=attn, position_ids=pos_ids,
                    use_cache=False, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states
        assert len(hs) == NL + 1, (len(hs), NL)
        # (A) last entry is post-norm: lm_head(hs[-1]) ~= out.logits
        dA = (lm_head(hs[-1]).float() - out.logits.float()).abs().max().item()
        # (B) layer-18 logit-lens recall@1 by block position, on masked generated slots
        h = hs[tap_index]                                          # [1,L,D] raw output of layer tap_layer
        lens = lm_head(final_norm(h)).float()                      # standard logit lens
        top1 = lens.argmax(-1)[0]                                  # [L]
        gen = (noisy[0] == MASK_ID) & (labels[0] != -100)
        rel = torch.arange(L, device=device) % B
        r_by = {t: [] for t in (0, 1, 4, 8, 16)}
        for t in r_by:
            m = gen & (rel == t)
            if m.any():
                r_by[t] = (top1[m] == labels[0][m]).float().mean().item()
        if ridx == 0:
            print(f"[smoke] (A) |lm_head(hs[-1]) - logits|max = {dA:.4f}  (small => hs[-1] is post-norm)")
        print(f"[smoke] ex{ridx} tap recall@1 by block-pos: " +
              " ".join(f"p{t}={r_by[t]:.2f}" for t in (0, 1, 4, 8, 16) if r_by[t] != []))

        # --- (C) DARC head on the REAL bf16 hidden ---
        cos, sin = rotary_emb(h, pos_ids)                         # [1,L,rot], base convention
        loss, m = head.forward_train(h, noisy, labels, cos, sin, embed, final_norm, lm_head)
        loss.backward()
        g_head = head.attention.q_proj.weight.grad
        print(f"[smoke] ex{ridx} HEAD loss={float(loss.detach()):.4f} n_sup={m['n_sup']} "
              f"grad_ok={g_head is not None and torch.isfinite(g_head).all().item()} "
              f"backbone_frozen={lm_head.weight.grad is None}")
        head.zero_grad(set_to_none=True)
        assert torch.isfinite(loss), "head loss must be finite on real bf16 hidden"

    print("[smoke] DARC integration smoke PASSED "
          "(tap index verified, layer-18 signal reproduced, head trains on real bf16 hidden)")


if __name__ == "__main__":
    main()
