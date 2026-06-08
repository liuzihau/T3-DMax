"""Block 4 — anchor-free top-K talk: GSM8K decode + eval.

Scores a trained top-K talk against the full-DMax baseline. Decode paradigm
(matches T3D_TOPK_TALK_PLAN.md):
  per response block:
    * THINK (full 20-layer) runs ONCE on [committed prefix + all-masked block]
      -> per-position top-K candidates for the block.
    * TALK (10-layer) iterates: still-masked positions are fed think's top-K
      soft-embedding (keep_mask_residual=True, inference variant); committed
      positions keep their token embedding. Commit at threshold 0.3 (DMax
      decode_uniform rule), stop at 0.9 / no-change / block full.

Compute: think_forwards (20-layer) + talk_forwards (10-layer). The 20-layer-
equivalent cost = think_fwd + 0.5*talk_fwd, vs full DMax = iters*1.0.

Run:
  python -m tasks.t3d_topk_eval_gsm8k \
    --think_path ../DMax-Math-16B-moe-merge \
    --talk_path  ./t3_topk_talk_merged_only_outputs/<hf_ckpt>  (or ../merged_10L untrained baseline) \
    --gen_length 512 --block_length 32 --threshold 0.3 --top_k 10 --limit 200
"""

from __future__ import annotations

import argparse
import re
import time

import torch
import torch.nn.functional as F

from tasks.t3d_topk_talk import load_causal_lm
from tasks.t3d_topk_soft_embed import build_topk_soft_embeds

GSM8K_USER_TEMPLATE = "Question: {question}\nLet's think step by step\nAnswer:"


# ---- proven decode helpers (copied from dinfer.decoding.generate_t3d) ----------
def build_block_causal_mask(L, block_length, dtype, device):
    idx = torch.arange(L, device=device)
    q_block = (idx // block_length).unsqueeze(1)
    kv_block = (idx // block_length).unsqueeze(0)
    allowed = (kv_block <= q_block)
    mask = torch.zeros(1, 1, L, L, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


def dmax_commit_uniform(logits, mask_index, active_index, threshold):
    """Left-to-right high-confidence prefix commit + 0.9 early-stop (DMax rule)."""
    x0 = logits.argmax(dim=-1)
    probs = F.softmax(logits.float(), dim=-1)
    max_probs = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    confidence = torch.where(mask_index, max_probs, torch.full_like(max_probs, -float("inf")))
    is_low_conf = mask_index & (confidence < threshold)
    has_failed = torch.cumsum(is_low_conf.long(), dim=1) > 0
    candidates = mask_index & (~has_failed)
    batch_has_sel = candidates.any(dim=-1, keepdim=True)
    mask_cumsum = torch.cumsum(mask_index.long(), dim=1)
    first_mask = (mask_cumsum == 1) & mask_index
    high_conf = torch.where(batch_has_sel, candidates, first_mask)
    breakflag = bool(active_index.any() and (max_probs[active_index] >= 0.9).all().item())
    return x0, high_conf, max_probs, breakflag


# ---- the anchor-free top-K decode ---------------------------------------------
@torch.no_grad()
def decode_topk_talk(think, talk, emb, mask_id, prompt_ids, *, gen_length, block_length,
                     threshold, top_k, max_iters, device, dtype, keep_mask_residual=True):
    P = prompt_ids.shape[1]
    L = ((P + gen_length + block_length - 1) // block_length) * block_length
    x = torch.full((1, L), mask_id, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids
    attn = build_block_causal_mask(L, block_length, dtype, device)
    pos = torch.arange(L, device=device).unsqueeze(0)
    first_b, num_b = P // block_length, L // block_length
    think_fwd = talk_fwd = 0

    for b in range(first_b, num_b):
        bs, be = b * block_length, (b + 1) * block_length
        m, p = attn[:, :, :be, :be], pos[:, :be]
        # THINK once on the all-masked block -> top-K for [bs, be)
        think_logits = think(inputs_embeds=emb(x[:, :be]), attention_mask=m,
                             position_ids=p, use_cache=False, return_dict=True).logits
        think_fwd += 1
        think_soft = build_topk_soft_embeds(think_logits[:, bs:be], emb, mask_id,
                                            top_k=top_k, keep_mask_residual=keep_mask_residual)   # [1, blk, D]
        # TALK iterates
        for _ in range(max_iters):
            block_x = x[:, bs:be]
            mask_index = (block_x == mask_id)
            if not bool(mask_index.any()):
                break
            inp = emb(x[:, :be]).clone()                       # committed -> token embed
            inp[:, bs:be][mask_index] = think_soft[mask_index].to(inp.dtype)   # masked -> think top-K
            talk_logits = talk(inputs_embeds=inp, attention_mask=m, position_ids=p,
                               use_cache=False, return_dict=True).logits[:, bs:be]
            talk_fwd += 1
            x0, high_conf, _, breakflag = dmax_commit_uniform(talk_logits, mask_index, mask_index, threshold)
            x[:, bs:be] = torch.where(high_conf, x0, block_x)
            if breakflag:
                break
    return x[:, P:P + gen_length], think_fwd, talk_fwd


# ---- GSM8K grading ------------------------------------------------------------
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _clean_num(s):
    return s.replace(",", "").rstrip(".") if s else s


def gold_answer(ans_text):
    m = re.search(r"####\s*(-?[\d,\.]+)", ans_text)
    return _clean_num(m.group(1)) if m else None


def pred_answer(text):
    nums = _NUM.findall(text)
    return _clean_num(nums[-1]) if nums else None


def is_correct(pred, gold):
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return pred == gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--think_path", required=True)
    ap.add_argument("--talk_path", required=True, help="trained talk hf ckpt (or ../merged_10L for the untrained baseline)")
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--gen_length", type=int, default=512)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--max_iters", type=int, default=32)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--debug_print", action="store_true")
    ap.add_argument("--no_mask_residual", action="store_true",
                    help="Feed the talk no-mask top-K (matches training keep_mask_residual=False). "
                         "Try this first — the mask-residual default is off-distribution for the talk.")
    args = ap.parse_args()
    dtype = torch.bfloat16

    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path or args.think_path, trust_remote_code=True)
    print("[eval] loading think + talk…")
    think = load_causal_lm(args.think_path, args.device, dtype)
    talk = load_causal_lm(args.talk_path, args.device, dtype)
    emb = think.get_input_embeddings()
    mask_id = 156895

    rows = [{"question": r["question"], "answer": r["answer"]}
            for r in load_dataset("gsm8k", "main", split="test")]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[eval] {len(rows)} problems  gen={args.gen_length} block={args.block_length} top_k={args.top_k}")

    n_ok = tot_think = tot_talk = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        messages = [{"role": "user", "content": GSM8K_USER_TEMPLATE.format(question=row["question"])}]
        prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=True, return_tensors="pt").to(args.device)
        resp, th, tk = decode_topk_talk(think, talk, emb, mask_id, prompt_ids,
                                        gen_length=args.gen_length, block_length=args.block_length,
                                        threshold=args.threshold, top_k=args.top_k,
                                        max_iters=args.max_iters, device=args.device, dtype=dtype,
                                        keep_mask_residual=not args.no_mask_residual)
        text = tok.decode(resp[0], skip_special_tokens=True)
        gold, pred = gold_answer(row["answer"]), pred_answer(text)
        ok = is_correct(pred, gold)
        n_ok += ok; tot_think += th; tot_talk += tk
        if args.debug_print or (i < 3):
            print(f"[{i}] {'OK ' if ok else 'XX '} pred={pred} gold={gold} think={th} talk={tk} tail={text[-120:]!r}")
        if (i + 1) % 25 == 0:
            print(f"  …{i+1}/{len(rows)}  acc={n_ok/(i+1):.3f}")

    n = len(rows)
    cost20 = tot_think + 0.5 * tot_talk            # 20-layer-equivalent forwards
    print("=" * 70)
    print(f"[eval] GSM8K acc = {n_ok}/{n} = {n_ok/n:.3f}")
    print(f"[eval] mean think/ex={tot_think/n:.1f}  talk/ex={tot_talk/n:.1f}  "
          f"20L-equiv/ex={cost20/n:.1f}  (full DMax ≈ iters*1.0)")
    print(f"[eval] {time.time()-t0:.0f}s. Baseline to beat: 84% @ gen512.")


if __name__ == "__main__":
    main()
