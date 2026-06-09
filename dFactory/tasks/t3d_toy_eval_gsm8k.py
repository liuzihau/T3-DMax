"""Toy / elementary GSM8K eval — per-block, oracle-prefix comparison.

Goal: with NO cascade contamination, compare two minimal-compute decode probes on
each answer block against the fully-converged DMax tokens, and expose block bias.

Per GSM8K example we sweep blocks left-to-right. At each block b the prefix (blocks
< b) is already the FULL-DMax-CONVERGED tokens (clean), and block b is all-MASK. On
that block we run three things, then commit the reference to form the next prefix:

  A) T3-D one-shot   : THINK once -> feed top-K to TALK once -> commit the confident
                       left-to-right prefix (argmax until first below threshold).
                       Cost = 1 think + 1 talk.
  B) DMax 2-iter     : THINK only, <=2 passes, each commits argmax>threshold (DMax
                       decode_uniform). Cost = up to 2 think.
  REF) DMax converged: THINK only to convergence (<= ref_max_iters). The "final
                       converged tokens"; also becomes the clean prefix for block b+1.

A and B run on the converged prefix but DO NOT write back (oracle-prefix probe);
only REF is committed into the running sequence.

Reported:
  * GSM8K acc for REF (real), A and B (oracle-prefix).
  * Per-block table: coverage (committed / answer positions) and precision
    (committed tokens == REF) for A and B  -> block bias.
  * Uncommitted positions are left masked (decode only what was committed).

Simplifications (toy): hard sticky commits (token embeds, like decode_topk_talk),
not DMax soft-embed feedback.

Run:
  python -m tasks.t3d_toy_eval_gsm8k \
    --think_path ../DMax-Math-16B-moe-merge \
    --talk_path  ./t3_topk_stage2_onpolicy_outputs/checkpoints/<step>/hf_ckpt \
    --gen_length 512 --block_length 32 --threshold 0.3 --top_k 10 --limit 20
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import torch

from tasks.t3d_topk_talk import load_causal_lm
from tasks.t3d_topk_soft_embed import build_topk_soft_embeds
from tasks.t3d_topk_eval_gsm8k import (
    GSM8K_USER_TEMPLATE,
    build_block_causal_mask,
    dmax_commit_uniform,
    gold_answer,
    pred_answer,
    is_correct,
)

MASK_ID = 156895


@torch.no_grad()
def _think_logits(model, emb, x, be, m, p):
    return model(inputs_embeds=emb(x[:, :be]), attention_mask=m, position_ids=p,
                 use_cache=False, return_dict=True).logits


@torch.no_grad()
def dmax_decode_block(think, emb, x, bs, be, m, p, threshold, max_iters):
    """Run THINK as a DMax (decode_uniform) decode on block b, IN PLACE on x, up to
    max_iters passes. Returns (#think forwards used)."""
    n = 0
    for _ in range(max_iters):
        block_x = x[:, bs:be]
        mask_index = (block_x == MASK_ID)
        if not bool(mask_index.any()):
            break
        logits = _think_logits(think, emb, x, be, m, p)[:, bs:be]
        n += 1
        x0, high_conf, _, breakflag = dmax_commit_uniform(logits, mask_index, mask_index, threshold)
        x[:, bs:be] = torch.where(high_conf, x0, block_x)
        if breakflag:
            break
    return n


@torch.no_grad()
def t3d_oneshot_block(think, talk, emb, x, bs, be, m, p, threshold, top_k, keep_mask_residual):
    """A: one THINK + one TALK on block b (reads x, does NOT write). Returns
    (argmax tokens [1,B], committed mask [1,B] over the block)."""
    mask_index = (x[:, bs:be] == MASK_ID)
    think_logits = _think_logits(think, emb, x, be, m, p)
    soft = build_topk_soft_embeds(think_logits[:, bs:be], emb, MASK_ID,
                                  top_k=top_k, keep_mask_residual=keep_mask_residual)
    inp = emb(x[:, :be]).clone()                       # committed prefix -> token embed
    inp[:, bs:be][mask_index] = soft[mask_index].to(inp.dtype)   # masked -> think top-K
    talk_logits = talk(inputs_embeds=inp, attention_mask=m, position_ids=p,
                       use_cache=False, return_dict=True).logits[:, bs:be]
    x0, high_conf, _, _ = dmax_commit_uniform(talk_logits, mask_index, mask_index, threshold)
    return x0, high_conf


def _decode_committed(tok, ids_row):
    keep = [int(t) for t in ids_row.tolist() if int(t) != MASK_ID]
    return tok.decode(keep, skip_special_tokens=True)


@torch.no_grad()
def sweep_example(think, talk, emb, prompt_ids, *, gen_length, block_length, threshold,
                  top_k, ref_max_iters, device, dtype, keep_mask_residual):
    """One example. Returns dict with REF/A/B assembled answers + per-block stats."""
    P = prompt_ids.shape[1]
    L = ((P + gen_length + block_length - 1) // block_length) * block_length
    x = torch.full((1, L), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids
    attn = build_block_causal_mask(L, block_length, dtype, device)
    pos = torch.arange(L, device=device).unsqueeze(0)
    first_b, num_b = P // block_length, L // block_length

    ansA = torch.full((1, L), MASK_ID, dtype=torch.long, device=device)
    ansB = torch.full((1, L), MASK_ID, dtype=torch.long, device=device)
    per_block = []   # list of (block_idx, statsA, statsB); stats = (committed, correct, total)
    th_fwd = tk_fwd = 0

    for b in range(first_b, num_b):
        bs, be = b * block_length, (b + 1) * block_length
        m, p = attn[:, :, :be, :be], pos[:, :be]
        answer_pos = (x[:, bs:be] == MASK_ID)              # masked answer positions in this block
        if not bool(answer_pos.any()):
            continue

        # ---- A: 1 think + 1 talk (oracle prefix, no write-back) ----
        a_tok, a_comm = t3d_oneshot_block(think, talk, emb, x, bs, be, m, p,
                                          threshold, top_k, keep_mask_residual)
        th_fwd += 1; tk_fwd += 1

        # ---- B: <=2 DMax think iters (on a copy) ----
        xB = x.clone()
        th_fwd += dmax_decode_block(think, emb, xB, bs, be, m, p, threshold, max_iters=2)
        b_tok = xB[:, bs:be]
        b_comm = answer_pos & (b_tok != MASK_ID)

        # ---- REF: full DMax convergence (writes into x -> clean prefix for b+1) ----
        th_fwd += dmax_decode_block(think, emb, x, bs, be, m, p, threshold, max_iters=ref_max_iters)
        ref_tok = x[:, bs:be]

        # ---- assemble committed answers + per-block stats vs converged REF ----
        ansA[:, bs:be] = torch.where(a_comm, a_tok, torch.full_like(a_tok, MASK_ID))
        ansB[:, bs:be] = torch.where(b_comm, b_tok, torch.full_like(b_tok, MASK_ID))
        total = int(answer_pos.sum())
        sA = (int(a_comm.sum()), int(((a_tok == ref_tok) & a_comm).sum()), total)
        sB = (int(b_comm.sum()), int(((b_tok == ref_tok) & b_comm).sum()), total)
        per_block.append((b, sA, sB))

    return {
        "ref": x[:, P:P + gen_length], "A": ansA[:, P:P + gen_length],
        "B": ansB[:, P:P + gen_length], "per_block": per_block,
        "think_fwd": th_fwd, "talk_fwd": tk_fwd,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--think_path", required=True)
    ap.add_argument("--talk_path", required=True, help="trained talk hf ckpt (or merged_10L baseline)")
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--gen_length", type=int, default=512)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--ref_max_iters", type=int, default=0, help="REF convergence cap; 0 => block_length")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--quiet", action="store_true", help="suppress per-sample text print")
    ap.add_argument("--tail", type=int, default=160, help="chars of decoded tail to print per method")
    ap.add_argument("--no_mask_residual", action="store_true",
                    help="feed talk no-mask top-K (match training keep_mask_residual=False)")
    args = ap.parse_args()
    dtype = torch.bfloat16
    ref_max_iters = args.ref_max_iters or args.block_length

    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path or args.think_path, trust_remote_code=True)
    print("[toy] loading think + talk…")
    think = load_causal_lm(args.think_path, args.device, dtype)
    talk = load_causal_lm(args.talk_path, args.device, dtype)
    emb = think.get_input_embeddings()

    rows = [{"question": r["question"], "answer": r["answer"]}
            for r in load_dataset("gsm8k", "main", split="test")][:args.limit]
    print(f"[toy] {len(rows)} problems  gen={args.gen_length} block={args.block_length} "
          f"top_k={args.top_k} thr={args.threshold} keep_mask_residual={not args.no_mask_residual}")

    ok = {"ref": 0, "A": 0, "B": 0}
    blk = defaultdict(lambda: {"A": [0, 0, 0], "B": [0, 0, 0]})   # block_idx -> committed/correct/total
    tot_think = tot_talk = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        messages = [{"role": "user", "content": GSM8K_USER_TEMPLATE.format(question=row["question"])}]
        prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=True, return_tensors="pt").to(args.device)
        out = sweep_example(think, talk, emb, prompt_ids,
                            gen_length=args.gen_length, block_length=args.block_length,
                            threshold=args.threshold, top_k=args.top_k,
                            ref_max_iters=ref_max_iters, device=args.device, dtype=dtype,
                            keep_mask_residual=not args.no_mask_residual)
        gold = gold_answer(row["answer"])
        texts = {k: _decode_committed(tok, out[k][0]) for k in ("ref", "A", "B")}
        preds = {k: pred_answer(texts[k]) for k in texts}
        res = {k: is_correct(preds[k], gold) for k in preds}
        for k in ok:
            ok[k] += res[k]
        for b, sA, sB in out["per_block"]:
            for j in range(3):
                blk[b]["A"][j] += sA[j]; blk[b]["B"][j] += sB[j]
        tot_think += out["think_fwd"]; tot_talk += out["talk_fwd"]

        if not args.quiet:
            print(f"\n=== [{i}] gold={gold}  "
                  f"REF={'OK' if res['ref'] else 'XX'}({preds['ref']}) "
                  f"A={'OK' if res['A'] else 'XX'}({preds['A']}) "
                  f"B={'OK' if res['B'] else 'XX'}({preds['B']}) ===")
            print(f"  Q: {row['question'][:140]!r}")
            for k in ("ref", "A", "B"):
                print(f"  {k:>3}: …{texts[k][-args.tail:]!r}")

    n = len(rows)
    print("\n" + "=" * 78)
    print(f"[toy] GSM8K acc   REF={ok['ref']}/{n}={ok['ref']/n:.3f}   "
          f"A(1think+1talk)={ok['A']}/{n}={ok['A']/n:.3f}   "
          f"B(DMax-2iter)={ok['B']}/{n}={ok['B']/n:.3f}   (oracle-prefix for A/B)")
    print(f"[toy] mean think/ex={tot_think/n:.1f}  talk/ex={tot_talk/n:.1f}")

    # per-block bias table (committed-token agreement vs converged REF)
    print("\n[toy] per-block committed-vs-converged  (cov=committed/answer, prec=correct/committed)")
    print(f"  {'blk':>4} | {'A cov':>6} {'A prec':>7} | {'B cov':>6} {'B prec':>7}")
    aggA = [0, 0, 0]; aggB = [0, 0, 0]
    for b in sorted(blk):
        cA, kA, tA = blk[b]["A"]; cB, kB, tB = blk[b]["B"]
        aggA = [aggA[0] + cA, aggA[1] + kA, aggA[2] + tA]
        aggB = [aggB[0] + cB, aggB[1] + kB, aggB[2] + tB]
        covA = cA / tA if tA else 0; precA = kA / cA if cA else 0
        covB = cB / tB if tB else 0; precB = kB / cB if cB else 0
        print(f"  {b:>4} | {covA:>6.2f} {precA:>7.2f} | {covB:>6.2f} {precB:>7.2f}")
    cA, kA, tA = aggA; cB, kB, tB = aggB
    print(f"  {'ALL':>4} | {(cA/tA if tA else 0):>6.2f} {(kA/cA if cA else 0):>7.2f} | "
          f"{(cB/tB if tB else 0):>6.2f} {(kB/cB if cB else 0):>7.2f}")
    print(f"[toy] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
