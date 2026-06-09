"""Toy / elementary GSM8K eval — per-block, oracle-prefix comparison.

Goal: with NO cascade contamination, compare two minimal-compute decode probes on
each answer block against the fully-converged DMax tokens, and expose block bias.

Per GSM8K example we sweep blocks left-to-right. At each block b the prefix (blocks
< b) is already the FULL-DMax-CONVERGED tokens (clean), and block b is all-MASK. We
run ONE shared THINK forward on the all-mask block, then:

  A) T3-D one-shot   : feed THINK's top-K to TALK once -> commit the confident
                       left-to-right prefix. Cost = 1 think + 1 talk.
  REF) DMax converged: THINK only, FAITHFUL DMax decode_uniform with soft-embedding
                       feedback (committed positions fed back as
                       p*embed(argmax)+(1-p)*embed(MASK), L2-renorm, rebuilt each
                       iter; mirrors dInfer generate_t3d). Runs to convergence and
                       becomes the clean prefix for block b+1. The "final converged
                       tokens" reference.
  B) DMax 2-iter     : NOT a separate decode — it is REF *snapshotted after 2 think
                       forwards*. Same procedure, so B's committed positions are a
                       subset of REF's (and may still shift as REF continues, since
                       DMax re-argmaxes committed positions each pass).

A runs on the converged prefix but does NOT write back (oracle-prefix probe); only
REF mutates the running sequence. The shared iter-0 THINK forward serves both A's
top-K hand-off and REF's first commit.

Reported:
  * GSM8K acc for REF (real), A and B (oracle-prefix).
  * Per-block table: coverage (committed / answer positions) and precision
    (committed tokens == REF) for A and B  -> block bias.
  * Uncommitted positions are left masked (decode only what was committed).

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
import torch.nn.functional as F

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


# --- DMax committed soft-embed feedback (copied from dInfer generate_t3d.build_inputs_embeds) ---
def build_inputs_embeds(logits, max_probs, x0, hard_token_ids, embedding_layer, mask_id,
                        committed_mask, uncommitted_mask, top_k=1):
    """Committed -> p*embed(argmax)+(1-p)*embed(MASK), L2-renorm; uncommitted -> embed(MASK).
    Stateless, rebuilt each iter. Returns block-shaped embeds [1, B, D]."""
    device = logits.device
    dtype = embedding_layer.weight.dtype
    base_embeds = embedding_layer(hard_token_ids).clone()
    if not bool(committed_mask.any()):
        return base_embeds
    if top_k == 1:
        topk_probs = max_probs.unsqueeze(-1)
        topk_indices = x0.unsqueeze(-1)
    else:
        probs = F.softmax(logits.float(), dim=-1)
        topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
    residual_probs = (1.0 - topk_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
    topk_embeds = embedding_layer(topk_indices).to(torch.float32)
    mask_embed = embedding_layer(torch.tensor([mask_id], device=device, dtype=torch.long)).to(torch.float32)
    mask_norm = mask_embed.norm(p=2)
    topk_weighted = (topk_embeds * topk_probs.unsqueeze(-1)).sum(dim=2)
    soft_embeds = topk_weighted + mask_embed.view(1, 1, -1) * residual_probs
    current_norm = soft_embeds.norm(p=2, dim=-1, keepdim=True)
    topk_norms = topk_embeds.norm(p=2, dim=-1)
    target_norm = (topk_norms * topk_probs).sum(dim=-1, keepdim=True) + mask_norm * residual_probs
    soft_embeds = (soft_embeds * (target_norm / (current_norm + 1e-6))).to(dtype)
    base_embeds[committed_mask] = soft_embeds[committed_mask]
    return base_embeds


@torch.no_grad()
def _dmax_commit_step(block_logits, x, bs, be, answer_index, emb_layer, threshold, soft_top_k):
    """One DMax decode_uniform commit step on the current block logits. Mutates x[block].
    Returns (done, soft_block_embeds, committed_mask). Mirrors generate_t3d._commit_and_build_embeds
    (think-only; no anchor)."""
    mask_idx = (x[:, bs:be] == MASK_ID)
    x0, high_conf, max_probs, breakflag = dmax_commit_uniform(block_logits, mask_idx, answer_index, threshold)
    update_mask = high_conf | (answer_index & ~mask_idx)          # re-argmax committed too (DMax)
    changed = update_mask & (x0 != x[:, bs:be])
    if bool(update_mask.any()):
        nb = x[0, bs:be].clone(); nb[update_mask[0]] = x0[0][update_mask[0]]; x[0, bs:be] = nb
    new_mask = (x[:, bs:be] == MASK_ID)
    committed = answer_index & (~new_mask)
    uncommitted = answer_index & new_mask
    soft = build_inputs_embeds(block_logits, max_probs, x0, x[:, bs:be], emb_layer, MASK_ID,
                               committed, uncommitted, top_k=soft_top_k)
    done = bool(breakflag) or (not bool(changed.any())) or (not bool(new_mask.any()))
    return done, soft, committed


@torch.no_grad()
def dmax_ref_decode(think, emb_layer, x, bs, be, m, p, answer_index, init_block_logits, *,
                    threshold, max_iters, soft_top_k, snapshot_after):
    """Faithful DMax decode of block b (THINK only, soft-embed feedback), IN PLACE on x,
    starting from init_block_logits (the shared iter-0 forward). Returns
    (extra_think_forwards, snap_tok [1,B], snap_committed [1,B]); snap_* = the block state
    after `snapshot_after` total think forwards (or convergence, whichever first) = method B."""
    fwd = 1                                                       # the shared iter-0 forward
    done, soft, committed = _dmax_commit_step(init_block_logits, x, bs, be, answer_index,
                                              emb_layer, threshold, soft_top_k)
    snap_tok = snap_comm = None
    if fwd >= snapshot_after or done:
        snap_tok, snap_comm = x[:, bs:be].clone(), committed.clone()
    extra = 0
    logits = init_block_logits
    while not done and fwd < max_iters:
        inp = emb_layer(x[:, :be]).clone()
        inp[:, bs:be] = soft.to(inp.dtype)
        logits = think(inputs_embeds=inp, attention_mask=m, position_ids=p,
                       use_cache=False, return_dict=True).logits[:, bs:be]
        fwd += 1; extra += 1
        done, soft, committed = _dmax_commit_step(logits, x, bs, be, answer_index,
                                                  emb_layer, threshold, soft_top_k)
        if snap_tok is None and (fwd >= snapshot_after or done):
            snap_tok, snap_comm = x[:, bs:be].clone(), committed.clone()
    still = (x[:, bs:be] == MASK_ID)                             # safety: never leak a MASK
    if bool(still.any()):
        fill = logits[0].argmax(-1)
        nb = x[0, bs:be].clone(); nb[still[0]] = fill[still[0]]; x[0, bs:be] = nb
    if snap_tok is None:
        snap_tok, snap_comm = x[:, bs:be].clone(), (answer_index & (x[:, bs:be] != MASK_ID))
    return extra, snap_tok, snap_comm


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
    per_block = []
    th_fwd = tk_fwd = 0

    for b in range(first_b, num_b):
        bs, be = b * block_length, (b + 1) * block_length
        m, p = attn[:, :, :be, :be], pos[:, :be]
        answer_pos = (x[:, bs:be] == MASK_ID)
        if not bool(answer_pos.any()):
            continue

        # ---- shared iter-0 THINK forward on the all-mask block ----
        think_logits0 = think(inputs_embeds=emb(x[:, :be]), attention_mask=m, position_ids=p,
                              use_cache=False, return_dict=True).logits[:, bs:be]
        th_fwd += 1

        # ---- A: 1 talk on think's top-K (oracle prefix, no write-back) ----
        soft = build_topk_soft_embeds(think_logits0, emb, MASK_ID, top_k=top_k,
                                      keep_mask_residual=keep_mask_residual)
        inp = emb(x[:, :be]).clone()
        inp[:, bs:be][answer_pos] = soft[answer_pos].to(inp.dtype)
        talk_logits = talk(inputs_embeds=inp, attention_mask=m, position_ids=p,
                           use_cache=False, return_dict=True).logits[:, bs:be]
        tk_fwd += 1
        a_tok, a_comm, _, _ = dmax_commit_uniform(talk_logits, answer_pos, answer_pos, threshold)

        # ---- REF (faithful DMax) + B snapshot (writes x -> clean prefix for b+1) ----
        extra, b_tok, b_comm = dmax_ref_decode(think, emb, x, bs, be, m, p, answer_pos, think_logits0,
                                               threshold=threshold, max_iters=ref_max_iters,
                                               soft_top_k=1, snapshot_after=2)
        th_fwd += extra
        ref_tok = x[:, bs:be]

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
    ap.add_argument("--ref_max_iters", type=int, default=0, help="REF convergence cap (forwards); 0 => block_length")
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
    blk = defaultdict(lambda: {"A": [0, 0, 0], "B": [0, 0, 0]})
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
