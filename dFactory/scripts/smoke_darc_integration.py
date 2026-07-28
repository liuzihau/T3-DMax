# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# DARC integration SMOKE (real model, 1-2 prompts, NO training harness / NO FSDP). Validates the risky core
# of the base-model integration before building train_darc.py:
#   (A) tap index: we tap hidden_states[tap_hidden_index] directly (== plot 'L{i}'; hs[i]=decoder layer i-1 out);
#       the last hidden_states
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
    ap.add_argument("--tap_hidden_index", type=int, default=18,
                    help="hidden_states index to tap == probe-plot 'L{i}'; hs[i]=decoder layer (i-1) out")
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
    tap_index = args.tap_hidden_index                              # direct hidden_states index (plot 'L{i}')
    dt = embed.weight.dtype
    print(f"[smoke] layers={NL} D={D} V={V} tap hidden_states[{tap_index}] "
          f"(= decoder layer {tap_index-1} out, plot 'L{tap_index}') dtype={dt}")

    cfg = DarcConfig(hidden_size=D, num_attention_heads=model.config.num_attention_heads,
                     num_key_value_heads=model.config.num_key_value_heads, head_dim=model.config.head_dim,
                     intermediate_size=model.config.intermediate_size, vocab_size=V,
                     rotary_dim=getattr(model.config, "rotary_dim", 64), block_size=args.block_length,
                     tap_hidden_index=args.tap_hidden_index)
    setattr(cfg, "mask_token_id", MASK_ID)
    head = DarcHead(cfg).to(device=device, dtype=dt)
    head.train()

    B = args.block_length
    from collections import defaultdict

    def tap_top1(noisy, pos_ids, attn):
        out = model(input_ids=noisy, attention_mask=attn, position_ids=pos_ids,
                    use_cache=False, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[tap_index]
        return h, lm_head(final_norm(h)).float().argmax(-1)[0], out                # h[1,L,D], top1[L]

    hit_all, cnt_all = defaultdict(float), defaultdict(float)                       # by in-block rel-pos
    hit_rp, cnt_rp = defaultdict(float), defaultdict(float)
    did_head = did_A = False
    rows = load_task(args.task, limit=args.limit)
    for ridx, row in enumerate(rows):
        pid = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                      add_generation_prompt=True, tokenize=True, return_tensors="pt").to(device)
        P = pid.shape[1]
        gx, ge = get_gold(model, embed, lm_head, final_norm, pid, args.gen_length, args.block_length, V, device)
        ge = int(ge)
        if ge - P < B:
            continue
        L = gx.shape[0]                                                             # block-aligned decode length
        gold = gx.to(device)
        pos_ids = torch.arange(L, device=device)[None]
        attn = build_block_causal_mask(L, B, dtype=dt, device=device)
        rel = (torch.arange(L, device=device) % B)

        # (1) ALL-MASKED: one forward, mask the whole generated region; recall over ALL generated positions
        noisy_all = gold.clone(); noisy_all[P:] = MASK_ID
        _, top1_all, out_all = tap_top1(noisy_all[None], pos_ids, attn)
        if not did_A:
            dA = (lm_head(out_all.hidden_states[-1]).float() - out_all.logits.float()).abs().max().item()
            print(f"[smoke] (A) |lm_head(hs[-1]) - logits|max = {dA:.4f}  (small => hs[-1] is post-norm)")
            did_A = True
        for p in range(P, ge):
            t = int(rel[p]); hit_all[t] += float(top1_all[p] == gold[p]); cnt_all[t] += 1

        # (2) REVEAL-PRIOR (faithful probe reproduction): per generated block, reveal [0,bs), mask [bs,L)
        first = ((P + B - 1) // B) * B                                             # first full-block boundary >= P
        for bs in range(first, ge, B):
            be = min(bs + B, ge)
            noisy_rp = gold.clone(); noisy_rp[bs:] = MASK_ID
            h_rp, top1_rp, out_rp = tap_top1(noisy_rp[None], pos_ids, attn)
            for p in range(bs, be):
                t = int(rel[p]); hit_rp[t] += float(top1_rp[p] == gold[p]); cnt_rp[t] += 1

            # (C) run the DARC head ONCE on a real reveal-prior block (bf16), loss on that block
            if not did_head:
                labels = torch.full((L,), -100, dtype=torch.long, device=device)
                labels[bs:be] = gold[bs:be]
                cos, sin = rotary_emb(h_rp, pos_ids)
                loss, mtr = head.forward_train(h_rp, noisy_rp[None], labels[None], cos, sin,
                                               embed, final_norm, lm_head)
                loss.backward()
                g = head.attention.q_proj.weight.grad
                print(f"[smoke] (C) HEAD loss={float(loss.detach()):.4f} n_sup={mtr['n_sup']} "
                      f"grad_ok={g is not None and torch.isfinite(g).all().item()} "
                      f"backbone_frozen={lm_head.weight.grad is None}")
                head.zero_grad(set_to_none=True)
                assert torch.isfinite(loss)

                # (D) Loss-2 top-layer replay faithfulness: running layers[tap_index:] -> norm -> lm_head on the
                # SAME tapped hidden must reconstruct the model's real out.logits (validates the replay path).
                import sys as _sys
                _sys.path.insert(0, os.path.join(_T3, "dFactory", "tasks"))
                from train_darc import replay_top                       # noqa: E402
                with torch.no_grad():
                    lg = replay_top(model, h_rp, tap_index, attn, pos_ids, (cos, sin), final_norm, lm_head)
                    dD = (lg.float() - out_rp.logits.float()).abs().max().item()
                # and the fuse at init is identity (zero-init) -> Loss-2 output == base final output
                with torch.no_grad():
                    soft = head.forward_train(h_rp, noisy_rp[None], labels[None], cos, sin,
                                              embed, final_norm, lm_head, return_soft_embeds=True)[2]
                    fused = head.fuse(soft, h_rp[:, bs:be])
                    dfuse = (fused - h_rp[:, bs:be]).abs().max().item()
                print(f"[smoke] (D) |replay(h) - out.logits|max = {dD:.4f}  (small => faithful top-layer replay); "
                      f"|fuse(init) - h|max = {dfuse:.4f} (0 => zero-init fuse == identity)")
                did_head = True

    def curve(hit, cnt):
        tot_h = sum(hit.values()); tot_c = sum(cnt.values())
        ts = [t for t in (0, 1, 2, 4, 8, 12, 16, 24, 30) if cnt.get(t, 0) > 0]
        s = " ".join(f"p{t}={hit[t]/cnt[t]:.2f}" for t in ts)
        return f"avg={tot_h/max(tot_c,1):.3f}  {s}"

    print(f"[smoke] recall@1 by in-block pos, ALL-MASKED   (prior NOT committed): {curve(hit_all, cnt_all)}")
    print(f"[smoke] recall@1 by in-block pos, REVEAL-PRIOR (== probe forward-1): {curve(hit_rp, cnt_rp)}")
    print("[smoke] EXPECT: REVEAL-PRIOR avg ~probe (~0.4, p0~0.9 decaying); ALL-MASKED clearly lower.")


if __name__ == "__main__":
    main()
