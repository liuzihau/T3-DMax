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
    for p in model.parameters():                                  # FREEZE the backbone (only the head trains)
        p.requires_grad_(False)
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

    B = args.block_length

    def tap_forward(noisy, L, pos_ids, attn):
        out = model(input_ids=noisy, attention_mask=attn, position_ids=pos_ids,
                    use_cache=False, output_hidden_states=True, return_dict=True)
        return out.hidden_states, out.logits

    def recall_at_block(top1, gold, bs, be, ts=(0, 1, 4, 8, 16)):
        return {t: float((top1[bs + t] == gold[bs + t])) for t in ts if bs + t < be}

    rows = load_task(args.task, limit=args.limit)
    for ridx, row in enumerate(rows):
        pid = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                      add_generation_prompt=True, tokenize=True, return_tensors="pt").to(device)
        P = pid.shape[1]
        gx, ge = get_gold(model, embed, lm_head, final_norm, pid, args.gen_length, args.block_length, V, device)
        # active block = first block boundary >= P with a FULL generated block (be <= gold eos)
        bs = ((P + B - 1) // B) * B
        be = bs + B
        if be > int(ge):
            print(f"[smoke] ex{ridx}: gold too short for a full generated block (P={P} ge={int(ge)}), skip")
            continue
        L = ((int(ge) // B) + 1) * B
        gold = gx[:L].clone().to(device)                          # clean gold tokens [L]
        pos_ids = torch.arange(L, device=device)[None]
        attn = build_block_causal_mask(L, B, dtype=dt, device=device)

        # --- contrast the two setups on the SAME active block ---
        # (1) ALL-MASKED (the wrong setup): mask the whole generated region
        noisy_all = gold.clone(); noisy_all[P:] = MASK_ID
        hs_all, logits_all = tap_forward(noisy_all[None], L, pos_ids, attn)
        top1_all = lm_head(final_norm(hs_all[tap_index])).float().argmax(-1)[0]
        # (2) REVEAL-PRIOR (matches inference): reveal [0,bs) as gold, mask [bs,L)
        noisy_rp = gold.clone(); noisy_rp[bs:] = MASK_ID
        hs_rp, _ = tap_forward(noisy_rp[None], L, pos_ids, attn)
        h = hs_rp[tap_index]                                      # [1,L,D]
        top1_rp = lm_head(final_norm(h)).float().argmax(-1)[0]

        if ridx == 0:
            dA = (lm_head(hs_all[-1]).float() - logits_all.float()).abs().max().item()
            print(f"[smoke] (A) |lm_head(hs[-1]) - logits|max = {dA:.4f}  (small => hs[-1] is post-norm)")
        r_all = recall_at_block(top1_all, gold, bs, be)
        r_rp = recall_at_block(top1_rp, gold, bs, be)
        fmt = lambda d: " ".join(f"p{t}={d[t]:.2f}" for t in sorted(d))
        print(f"[smoke] ex{ridx} block[{bs},{be}) recall@1  ALL-MASKED: {fmt(r_all)}")
        print(f"[smoke] ex{ridx} block[{bs},{be}) recall@1  REVEAL-PRIOR: {fmt(r_rp)}   <- should be ~0.9 at p0")

        # --- (C) DARC head on the REAL bf16 hidden (reveal-prior), loss on the active block only ---
        labels = torch.full((L,), -100, dtype=torch.long, device=device)
        labels[bs:be] = gold[bs:be]                               # head skips the block seed (rel-pos 0) itself
        cos, sin = rotary_emb(h, pos_ids)
        loss, m = head.forward_train(h, noisy_rp[None], labels[None], cos, sin, embed, final_norm, lm_head)
        loss.backward()
        g_head = head.attention.q_proj.weight.grad
        print(f"[smoke] ex{ridx} HEAD loss={float(loss.detach()):.4f} n_sup={m['n_sup']} "
              f"grad_ok={g_head is not None and torch.isfinite(g_head).all().item()} "
              f"backbone_frozen={lm_head.weight.grad is None}")
        head.zero_grad(set_to_none=True)
        assert torch.isfinite(loss), "head loss must be finite on real bf16 hidden"

    print("[smoke] done — check that REVEAL-PRIOR p0 ~0.9 (reproduces the probe) and ALL-MASKED is low.")


if __name__ == "__main__":
    main()
