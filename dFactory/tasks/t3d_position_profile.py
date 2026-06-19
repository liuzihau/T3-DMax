"""B — within-block-position reliability profile (think vs talk, single forward).

Hypothesis (b): a shallow TALK has a SHORTER reliable decode window than the deep THINK. On a
fully-masked block (conditioned on clean earlier context), measure -- per within-block position
j -- each model's SINGLE-FORWARD argmax confidence and its agreement with THINK's CONVERGED decode
of that block (the best per-token truth proxy; GSM8K has no per-token gold). If talk's agreement
falls off a cliff at j~4-8 while think holds to j~12, (b) holds and retraining talk at a smaller
block_size is justified. If think and talk decay together, the gap is NOT a window effect.

  think_1 : think(masked block) ONCE                  -> per-pos argmax + confidence
  talk_1  : talk(masked block + think top-K soft) ONCE -> per-pos argmax + confidence   (infer-faithful)
  ref     : think decoded to CONVERGENCE               -> per-pos reference ids (truth proxy)

Per position j we report: think conf, think agree(=think_1==ref), talk conf, talk agree(=talk_1==ref),
and talk-vs-think single-pass agreement. Aggregated over the first generation block of `limit` prompts.
The "reliable window" = the largest prefix length where mean agreement stays >= --window_floor.

Run (GPU box):
  python -m tasks.t3d_position_profile \
      --think_path ../DMax-Math-16B-moe-merge \
      --talk_path  ./t3_topk_stage2_onpolicy_outputs/checkpoints/<step>/hf_ckpt \
      --block_length 32 --top_k 10 --threshold 0.3 --gen_block 1 --limit 50
"""

from __future__ import annotations

import argparse
import torch

from tasks.t3d_topk_talk import load_causal_lm
from tasks.t3d_topk_soft_embed import build_topk_soft_embeds
from tasks.t3d_topk_eval_gsm8k import build_block_causal_mask, GSM8K_USER_TEMPLATE
from tasks.t3d_probe_converged_teacher import think_converge

MASK_ID = 156895


def _conf(logits):
    """argmax id + its softmax prob, per position. logits [1, n, V] -> (ids [1,n], probs [1,n])."""
    p = logits.float().softmax(-1)
    a = p.argmax(-1)
    return a, p.gather(-1, a.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def profile_example(think, talk, emb, x, bs, be, B, m, p, top_k, threshold, max_iters, kmr, acc):
    """Accumulate per-position stats for one fully-masked target block [bs:be] into `acc`
    (a dict of length-B float tensors). Returns nothing; mutates `acc` in place."""
    masked = (x[:, bs:be] == MASK_ID)                          # [1, B]
    if not bool(masked.any()):
        return

    # think_1 -- one think forward on the masked block
    th1 = think(inputs_embeds=emb(x[:, :be]), attention_mask=m, position_ids=p,
                use_cache=False, return_dict=True).logits[:, bs:be]
    a1, c1 = _conf(th1)

    # talk_1 -- one talk forward, masked positions fed think's top-K soft-embed (inference-faithful)
    soft = build_topk_soft_embeds(th1, emb, MASK_ID, top_k=top_k, keep_mask_residual=kmr)
    inp = emb(x[:, :be]).clone()
    inp[:, bs:be][masked] = soft[masked].to(inp.dtype)
    ta1 = talk(inputs_embeds=inp, attention_mask=m, position_ids=p,
               use_cache=False, return_dict=True).logits[:, bs:be]
    aT, cT = _conf(ta1)

    # ref -- think decoded to convergence on a CLONE (the per-token truth proxy)
    xref = x.clone()
    _, ref_ids, _, _ = think_converge(think, emb, xref, bs, be, m, p, threshold, max_iters, top_k)

    for j in range(B):
        if not bool(masked[0, j]):
            continue
        acc["n"][j] += 1.0
        acc["think_conf"][j] += float(c1[0, j])
        acc["talk_conf"][j] += float(cT[0, j])
        acc["think_agree"][j] += float(a1[0, j] == ref_ids[0, j])
        acc["talk_agree"][j] += float(aT[0, j] == ref_ids[0, j])
        acc["talk_vs_think"][j] += float(aT[0, j] == a1[0, j])


def _window(curve, floor):
    """Largest prefix length L such that curve[0..L-1] all >= floor (the reliable window)."""
    L = 0
    for v in curve:
        if v != v or v < floor:                                 # NaN or below floor -> stop
            break
        L += 1
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--think_path", required=True)
    ap.add_argument("--talk_path", required=True, help="trained talk ckpt (or ../merged_10L untrained)")
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--max_iters", type=int, default=32)
    ap.add_argument("--gen_block", type=int, default=1,
                    help="first fully-masked GENERATION block to profile (>=1; conditioned on think-decoded "
                         "earlier blocks for clean context).")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--bin", type=int, default=4, help="group size for the binned summary table.")
    ap.add_argument("--window_floor", type=float, default=0.5,
                    help="agreement threshold defining the 'reliable window' (largest prefix >= floor).")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dtype = torch.bfloat16
    B = args.block_length
    kmr = True                                                  # inference always blends the talk soft with the mask residual

    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path or args.think_path, trust_remote_code=True)
    print("[profile] loading think + talk…")
    think = load_causal_lm(args.think_path, args.device, dtype)
    talk = load_causal_lm(args.talk_path, args.device, dtype)
    emb = think.get_input_embeddings()

    keys = ("n", "think_conf", "talk_conf", "think_agree", "talk_agree", "talk_vs_think")
    acc = {k: torch.zeros(B, dtype=torch.float64) for k in keys}

    rows = list(load_dataset("gsm8k", "main", split="test"))[:args.limit]
    print(f"[profile] {len(rows)} prompts  block={B} gen_block={args.gen_block} top_k={args.top_k}")
    for i, row in enumerate(rows):
        messages = [{"role": "user", "content": GSM8K_USER_TEMPLATE.format(question=row["question"])}]
        prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=True, return_tensors="pt").to(args.device)
        P = prompt_ids.shape[1]
        first_b = P // B
        target_b = first_b + args.gen_block
        L = (target_b + 1) * B
        x = torch.full((1, L), MASK_ID, dtype=torch.long, device=args.device)
        x[:, :P] = prompt_ids
        attn = build_block_causal_mask(L, B, dtype, args.device)
        pos = torch.arange(L, device=args.device).unsqueeze(0)
        # decode blocks before the target with think -> clean preceding context
        for b in range(first_b, target_b):
            be0 = (b + 1) * B
            think_converge(think, emb, x, b * B, be0, attn[:, :, :be0, :be0], pos[:, :be0],
                           args.threshold, args.max_iters, args.top_k)
        bs, be = target_b * B, (target_b + 1) * B
        profile_example(think, talk, emb, x, bs, be, B, attn[:, :, :be, :be], pos[:, :be],
                        args.top_k, args.threshold, args.max_iters, kmr, acc)
        if (i + 1) % 10 == 0:
            print(f"  …{i + 1}/{len(rows)}")

    n = acc["n"].clamp(min=1.0)
    th_conf = (acc["think_conf"] / n).tolist()
    tk_conf = (acc["talk_conf"] / n).tolist()
    th_agr = (acc["think_agree"] / n).tolist()
    tk_agr = (acc["talk_agree"] / n).tolist()
    tvt = (acc["talk_vs_think"] / n).tolist()
    cnt = acc["n"].tolist()

    print("=" * 78)
    print("[profile] per-position (agreement = single forward vs think-CONVERGED ref)")
    print(f"  {'pos':>3} | {'n':>5} | {'th_conf':>7} {'th_agree':>8} | {'tk_conf':>7} {'tk_agree':>8} | {'tk=th':>6}")
    for j in range(B):
        if cnt[j] == 0:
            continue
        print(f"  {j:>3} | {cnt[j]:>5.0f} | {th_conf[j]:>7.3f} {th_agr[j]:>8.3f} | "
              f"{tk_conf[j]:>7.3f} {tk_agr[j]:>8.3f} | {tvt[j]:>6.3f}")

    print("-" * 78)
    print(f"[profile] binned (size {args.bin}) mean agreement vs think-converged")
    print(f"  {'positions':>11} | {'think':>6} | {'talk':>6} | {'gap':>6}")
    for b0 in range(0, B, args.bin):
        b1 = min(B, b0 + args.bin)
        seg = [k for k in range(b0, b1) if cnt[k] > 0]
        if not seg:
            continue
        ta = sum(th_agr[k] for k in seg) / len(seg)
        la = sum(tk_agr[k] for k in seg) / len(seg)
        print(f"  {f'{b0:>2}-{b1 - 1:<2}':>11} | {ta:>6.3f} | {la:>6.3f} | {ta - la:>6.3f}")

    tw = _window(th_agr, args.window_floor)
    lw = _window(tk_agr, args.window_floor)
    print("-" * 78)
    print(f"[profile] reliable window (largest prefix with agreement >= {args.window_floor}):")
    print(f"          think = {tw} tokens   talk = {lw} tokens   (block = {B})")
    print("Read: talk window << think window CONFIRMS hypothesis (b) -> retrain talk at a smaller "
          "block_size. think window itself << block tells you think's per-pass tail is filler "
          "(refined over later passes). Agreement is fidelity-to-think, not gold-correctness.")


if __name__ == "__main__":
    main()
