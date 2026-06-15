"""Toy probe: is think's CONVERGED (K-iter) distribution a better KL teacher than think's
ONE-PASS (filler-flooded) distribution, and where does a single talk pass land?

For the FIRST generation block of a few GSM8K prompts (block fully masked):
  think_1  = think(X) ONCE on the masked block         -> the CURRENT teacher (filler tail expected)
  talk_1   = talk(X + think_1 top-K soft-embed) ONCE   -> one student pass (the inference dynamic)
  think_K  = think decoded K iters (DMax decode_uniform) -> the PROPOSED teacher (~ gold proxy)

Prints, per block position: think_1 / think_K / talk_1 argmax token + confidence, and agreement
flags; plus the decoded text of each and aggregate agreements. What to look for:
  * think_1 floods low-content tokens (\n, the) at high confidence beyond a short prefix  -> the poison.
  * think_K is clean content at most positions (the better target).
  * talk_1 argmax agrees more with think_1 (mimics the filler) than with think_K (the answer)
    -> confirms the talk is distilling think's deferral, and that think_K is the target to use.

Run (on the GPU box):
  python -m tasks.t3d_probe_converged_teacher \
      --think_path ../DMax-Math-16B-moe-merge \
      --talk_path  ./t3_topk_stage2_onpolicy_outputs/checkpoints/<step>/hf_ckpt \
      --block_length 32 --top_k 10 --threshold 0.3 --k_iters 4 --limit 5
"""

from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F

from tasks.t3d_topk_talk import load_causal_lm
from tasks.t3d_topk_soft_embed import build_topk_soft_embeds
from tasks.t3d_topk_eval_gsm8k import build_block_causal_mask, dmax_commit_uniform, GSM8K_USER_TEMPLATE

MASK_ID = 156895


def _conf(logits):
    """argmax id + its softmax prob, per position. logits [1, n, V] -> (ids [1,n], probs [1,n])."""
    p = logits.softmax(-1)
    a = p.argmax(-1)
    return a, p.gather(-1, a.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def think_k_iters(think, emb, x, bs, be, m, p, threshold, k_iters):
    """Run k DMax decode_uniform iters of think on block [bs:be] (committing think's own argmax),
    then one more forward -> the converged logits for the block + the committed token ids."""
    for _ in range(k_iters):
        block = x[:, bs:be]
        mask_index = block == MASK_ID
        if not bool(mask_index.any()):
            break
        logits = think(inputs_embeds=emb(x[:, :be]), attention_mask=m, position_ids=p,
                       use_cache=False, return_dict=True).logits
        x0, high_conf, _, brk = dmax_commit_uniform(logits[:, bs:be], mask_index, mask_index, threshold)
        x[:, bs:be] = torch.where(high_conf, x0, block)
        if brk:
            break
    final = think(inputs_embeds=emb(x[:, :be]), attention_mask=m, position_ids=p,
                  use_cache=False, return_dict=True).logits[:, bs:be]
    return final, x[:, bs:be].clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--think_path", required=True)
    ap.add_argument("--talk_path", required=True, help="trained talk ckpt (or ../merged_10L untrained)")
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--k_iters", type=int, default=4, help="extra think iterations for the converged teacher")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_mask_residual", action="store_true", help="match a talk trained with keep_mask_residual=false")
    args = ap.parse_args()
    dtype = torch.bfloat16

    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path or args.think_path, trust_remote_code=True)
    think = load_causal_lm(args.think_path, args.device, dtype)
    talk = load_causal_lm(args.talk_path, args.device, dtype)
    emb = think.get_input_embeddings()
    B = args.block_length
    rows = list(load_dataset("gsm8k", "main", split="test"))[:args.limit]

    agree_1 = agree_K = both = tot = 0
    for i, row in enumerate(rows):
        messages = [{"role": "user", "content": GSM8K_USER_TEMPLATE.format(question=row["question"])}]
        prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=True, return_tensors="pt").to(args.device)
        P = prompt_ids.shape[1]
        bs = (P // B) * B
        be = bs + B
        L = be
        x = torch.full((1, L), MASK_ID, dtype=torch.long, device=args.device)
        x[:, :P] = prompt_ids                                   # [bs, P) is prompt tail, [P, be) is masked
        attn = build_block_causal_mask(L, B, dtype, args.device)
        m, p = attn[:, :, :be, :be], torch.arange(L, device=args.device).unsqueeze(0)[:, :be]
        masked = (x[:, bs:be] == MASK_ID)                       # [1, B] the masked positions in this block

        # think_1 (one pass) + the top-K soft-embed it produces
        think1 = think(inputs_embeds=emb(x[:, :be]), attention_mask=m, position_ids=p,
                       use_cache=False, return_dict=True).logits[:, bs:be]
        a1, c1 = _conf(think1)
        soft = build_topk_soft_embeds(think1, emb, MASK_ID, top_k=args.top_k,
                                      keep_mask_residual=not args.no_mask_residual)
        # talk_1 (one pass) fed think_1's top-K at the masked positions
        inp = emb(x[:, :be]).clone()
        inp[:, bs:be][masked] = soft[masked].to(inp.dtype)
        talk1 = talk(inputs_embeds=inp, attention_mask=m, position_ids=p,
                     use_cache=False, return_dict=True).logits[:, bs:be]
        aT, cT = _conf(talk1)
        # think_K (converged) on a COPY of the masked state
        xk = x.clone()
        thinkK, thinkK_ids = think_k_iters(think, emb, xk, bs, be, m, p, args.threshold, args.k_iters)
        aK, cK = _conf(thinkK)

        # decoded blocks (only the originally-masked tail is interesting). a1/aK/aT and thinkK_ids are [1, B].
        def dec(ids):
            return tok.decode(ids[0, masked[0]], skip_special_tokens=False)
        print("=" * 100)
        print(f"[{i}] Q: {row['question'][:90]}…")
        print(f"  think_1  block (greedy): {dec(a1)!r}")
        print(f"  think_K  block (decoded): {dec(thinkK_ids)!r}")
        print(f"  talk_1   block (greedy): {dec(aT)!r}")
        # per-position table (masked positions only)
        print(f"  {'pos':>3} | {'think_1 (conf)':>22} | {'think_K (conf)':>22} | {'talk_1 (conf)':>22} | t1=tK t1=tT tK=tT")
        for j in range(B):
            if not bool(masked[0, j]):
                continue
            t1 = tok.decode([int(a1[0, j])]); tk = tok.decode([int(aK[0, j])]); tt = tok.decode([int(aT[0, j])])
            f1K = "Y" if int(a1[0, j]) == int(aK[0, j]) else "."
            f1T = "Y" if int(a1[0, j]) == int(aT[0, j]) else "."
            fKT = "Y" if int(aK[0, j]) == int(aT[0, j]) else "."
            print(f"  {bs+j:>3} | {repr(t1)[:14]:>14}({c1[0,j]:.2f}) | {repr(tk)[:14]:>14}({cK[0,j]:.2f}) | "
                  f"{repr(tt)[:14]:>14}({cT[0,j]:.2f}) |   {f1K}     {f1T}     {fKT}")
        # aggregate agreement on the masked positions
        mm = masked[0]
        a1m, aKm, aTm = a1[0, mm], aK[0, mm], aT[0, mm]
        n = int(mm.sum())
        a_talk_think1 = int((aTm == a1m).sum())
        a_talk_thinkK = int((aTm == aKm).sum())
        agree_1 += a_talk_think1; agree_K += a_talk_thinkK; tot += n
        print(f"  -> talk_1 argmax agrees with think_1 (filler): {a_talk_think1}/{n} = {a_talk_think1/max(1,n):.2f} ; "
              f"with think_K (converged): {a_talk_thinkK}/{n} = {a_talk_thinkK/max(1,n):.2f}")

    print("=" * 100)
    print(f"[TOTAL] talk_1 vs think_1 (one-pass/filler): {agree_1}/{tot} = {agree_1/max(1,tot):.3f}")
    print(f"[TOTAL] talk_1 vs think_K (converged target): {agree_K}/{tot} = {agree_K/max(1,tot):.3f}")
    print("If talk_1 tracks think_1 >> think_K, the talk is distilling think's DEFERRAL; "
          "use think_K (correctness-gated) as the teacher and/or gold-CE-only + calibration (label smoothing, higher tau).")


if __name__ == "__main__":
    main()
