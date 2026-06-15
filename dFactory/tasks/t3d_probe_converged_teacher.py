"""Toy probe: compare, on TWO consecutive generation blocks of a few GSM8K prompts, the per-position
predictions of several decode strategies, each split into the COMMIT region (the confident left-to-right
prefix decode_uniform would commit at tau) vs the STILL-MASK region (the deferred tail, where the
\\n/repetition filler collapse lives):

  think_1        = think(X) ONCE                                  (think's 1st pass)
  talk_1         = talk(X + think_1 top-K soft-embed) ONCE        (1 think + 1 talk = lightweight 2nd
                                                                   forward; reference = think_2)
  think_2        = think_1 -> commit think_1's prefix -> think    (think's REAL 2nd iteration; +N commits)
  talk_think_*   = think_1 -> talk_1 -> commit talk_1's prefix    (does think CONVERGE from the TALK's
                   -> think UNTIL CONVERGED                        commit? reports #iters)
  base_think     = think decoded ALONE UNTIL CONVERGED            (the pure-think baseline answer for the
                                                                   block; final tokens + per-pos confidence
                                                                   + #iters)

Two blocks per example: the SECOND block is conditioned on the BASELINE-THINK decode of the first block
(clean preceding context). Reads:
  * COMMIT region: think_1 / talk_1 / think_2 agree (easy frontier).
  * STILL-MASK region: think_1 floods filler; does talk_1 mimic it or match think_2? -> talk_1-vs-think_2
    STILL agreement is "can the lightweight 2nd forward leap one think iter on the hard tail?" (expect LOW).
  * talk_think_converged vs base_think: does starting from the TALK's commit reach the same place as pure
    think? (does think recover the talk's commit?)

Run (on the GPU box):
  python -m tasks.t3d_probe_converged_teacher \
      --think_path ../DMax-Math-16B-moe-merge \
      --talk_path  ./t3_topk_stage2_onpolicy_outputs/checkpoints/<step>/hf_ckpt \
      --block_length 32 --top_k 10 --threshold 0.6 --gen_block 1 --limit 5
"""

from __future__ import annotations

import argparse
import time
import torch

from tasks.t3d_topk_talk import load_causal_lm, confident_prefix_commit
from tasks.t3d_topk_soft_embed import build_topk_soft_embeds
from tasks.t3d_topk_eval_gsm8k import build_block_causal_mask, dmax_commit_uniform, GSM8K_USER_TEMPLATE

MASK_ID = 156895


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _conf(logits):
    """argmax id + its softmax prob, per position. logits [1, n, V] -> (ids [1,n], probs [1,n])."""
    p = logits.softmax(-1)
    a = p.argmax(-1)
    return a, p.gather(-1, a.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def think_converge(think, emb, x, bs, be, m, p, threshold, max_iters):
    """Run think's DMax decode_uniform on block [bs:be] UNTIL no masked positions remain (or max_iters).
    Mutates x[:, bs:be] in place (commits think's own argmax each iter). Returns
    (final_logits[block], committed_ids[block], n_iters)."""
    n = 0
    while bool((x[:, bs:be] == MASK_ID).any()) and n < max_iters:
        block = x[:, bs:be]
        mask_index = block == MASK_ID
        logits = think(inputs_embeds=emb(x[:, :be]), attention_mask=m, position_ids=p,
                       use_cache=False, return_dict=True).logits
        x0, high_conf, _, _ = dmax_commit_uniform(logits[:, bs:be], mask_index, mask_index, threshold)
        x[:, bs:be] = torch.where(high_conf, x0, block)
        n += 1
    final = think(inputs_embeds=emb(x[:, :be]), attention_mask=m, position_ids=p,
                  use_cache=False, return_dict=True).logits[:, bs:be]
    return final, x[:, bs:be].clone(), n


@torch.no_grad()
def mixed_converge(think, talk, emb, x, bs, be, m, p, threshold, top_k, kmr, max_iters, schedule):
    """Decode block [bs:be] UNTIL CONVERGED with a MIXED think/talk schedule: schedule(i) -> 'think'|'talk'
    for iteration i. A think step runs think on the committed state (bare [MASK] at undecided). A talk step
    runs the talk fed the top-K soft-embed of the PREVIOUS pass's logits (think's if the prev step was think,
    the talk's own otherwise -- the inference dynamic). Commits the confident prefix each step. Mutates
    x[:, bs:be]. Returns (final_logits[block], ids[block], n_think, n_talk)."""
    cand = None
    nth = ntk = 0
    while bool((x[:, bs:be] == MASK_ID).any()) and (nth + ntk) < max_iters:
        block = x[:, bs:be]
        mask_index = block == MASK_ID
        if schedule(nth + ntk) == "think":
            logits = think(inputs_embeds=emb(x[:, :be]), attention_mask=m, position_ids=p,
                           use_cache=False, return_dict=True).logits
            nth += 1
        else:
            src = cand if cand is not None else think(inputs_embeds=emb(x[:, :be]), attention_mask=m,
                                                      position_ids=p, use_cache=False, return_dict=True).logits
            soft = build_topk_soft_embeds(src[:, bs:be], emb, MASK_ID, top_k=top_k, keep_mask_residual=kmr)
            inp = emb(x[:, :be]).clone()
            inp[:, bs:be][mask_index] = soft[mask_index].to(inp.dtype)
            logits = talk(inputs_embeds=inp, attention_mask=m, position_ids=p,
                          use_cache=False, return_dict=True).logits
            ntk += 1
        cand = logits
        x0, high_conf, _, _ = dmax_commit_uniform(logits[:, bs:be], mask_index, mask_index, threshold)
        x[:, bs:be] = torch.where(high_conf, x0, block)
    final = think(inputs_embeds=emb(x[:, :be]), attention_mask=m, position_ids=p,
                  use_cache=False, return_dict=True).logits[:, bs:be]
    return final, x[:, bs:be].clone(), nth, ntk


@torch.no_grad()
def probe_block(think, talk, emb, tok, x, bs, be, B, m, p, args, agg, tag):
    """Analyse one block [bs:be] of x (preceding blocks already committed). Prints the 5 strategies
    split into commit/still regions + a per-position table. Returns base_think's committed ids [1,B]
    (so the caller can use them as the next block's context)."""
    masked = (x[:, bs:be] == MASK_ID)                       # [1, B] masked positions in this block
    if not bool(masked.any()):
        print(f"  ({tag}) block {bs}:{be} has no masked positions; skipping.")
        return x[:, bs:be].clone()

    def dtxt(ids_row, region_row):
        sel = ids_row[region_row]
        return tok.decode(sel, skip_special_tokens=False) if sel.numel() else "(empty)"

    # ---- think_1 (one pass) ----  [timed: the seed forward of the talk_think method]
    _sync(); _t = time.time()
    th1 = think(inputs_embeds=emb(x[:, :be]), attention_mask=m, position_ids=p,
                use_cache=False, return_dict=True).logits[:, bs:be]
    _sync(); t_think1 = time.time() - _t
    a1, c1 = _conf(th1)
    am1, cm1 = confident_prefix_commit(th1, masked, B, args.threshold)
    sm1 = masked & (~cm1)

    # ---- talk_1 (one think + one talk) ----
    soft = build_topk_soft_embeds(th1, emb, MASK_ID, top_k=args.top_k,
                                  keep_mask_residual=not args.no_mask_residual)
    inp = emb(x[:, :be]).clone()
    inp[:, bs:be][masked] = soft[masked].to(inp.dtype)
    _sync(); _t = time.time()
    ta1 = talk(inputs_embeds=inp, attention_mask=m, position_ids=p,
               use_cache=False, return_dict=True).logits[:, bs:be]
    _sync(); t_talk1 = time.time() - _t
    aT, cT = _conf(ta1)
    amT, cmT = confident_prefix_commit(ta1, masked, B, args.threshold)
    smT = masked & (~cmT)

    # ---- think_2 = think_1 -> commit think_1's prefix (hard) -> think ----
    x1 = x.clone()
    x1[:, bs:be] = torch.where(cm1, am1, x1[:, bs:be])
    th2 = think(inputs_embeds=emb(x1[:, :be]), attention_mask=m, position_ids=p,
                use_cache=False, return_dict=True).logits[:, bs:be]
    a2, c2 = _conf(th2)
    mask2 = (x1[:, bs:be] == MASK_ID)
    am2, cm2 = confident_prefix_commit(th2, mask2, B, args.threshold)
    sm2 = mask2 & (~cm2)

    # ---- talk_think_converged = think_1 -> talk_1 -> commit talk_1's prefix -> think until converged ----
    xt = x.clone()
    xt[:, bs:be] = torch.where(cmT, amT, xt[:, bs:be])
    _sync(); _t = time.time()
    ttk_logits, ttk_ids, ttk_iters = think_converge(think, emb, xt, bs, be, m, p, args.threshold, args.max_iters)
    _sync(); t_ttk = time.time() - _t
    atk, ctk = _conf(ttk_logits)
    ttk_commit = masked & (ttk_ids != MASK_ID)
    ttk_still = masked & (ttk_ids == MASK_ID)
    # ALL forwards of the talk_think method: 1 think (seed) + 1 talk + ttk_iters think (the converge loop)
    ttk_fwd = 1 + 1 + ttk_iters
    ttk_wall = t_think1 + t_talk1 + t_ttk

    # ---- base_think = pure think decoded UNTIL CONVERGED (no talk) ----
    xb = x.clone()
    _sync(); _t = time.time()
    base_logits, base_ids, base_iters = think_converge(think, emb, xb, bs, be, m, p, args.threshold, args.max_iters)
    _sync(); t_base = time.time() - _t
    ab, cb = _conf(base_logits)
    base_still = masked & (base_ids == MASK_ID)
    base_fwd = base_iters                                  # all think (the converge loop)

    # ---- method C: think + think + talk... UNTIL CONVERGED (2 think seeds, then talk) ----
    xc = x.clone()
    _sync(); _t = time.time()
    c_logits, c_ids, c_nth, c_ntk = mixed_converge(
        think, talk, emb, xc, bs, be, m, p, args.threshold, args.top_k, not args.no_mask_residual,
        args.max_iters, lambda i: "think" if i < 2 else "talk")
    _sync(); t_c = time.time() - _t
    ac, cc = _conf(c_logits)
    c_commit = masked & (c_ids != MASK_ID); c_still = masked & (c_ids == MASK_ID)

    # ---- method D: think + talk + think + talk + ... UNTIL CONVERGED (strictly alternating) ----
    xd = x.clone()
    _sync(); _t = time.time()
    d_logits, d_ids, d_nth, d_ntk = mixed_converge(
        think, talk, emb, xd, bs, be, m, p, args.threshold, args.top_k, not args.no_mask_residual,
        args.max_iters, lambda i: "think" if i % 2 == 0 else "talk")
    _sync(); t_d = time.time() - _t
    ad, cd = _conf(d_logits)
    d_commit = masked & (d_ids != MASK_ID); d_still = masked & (d_ids == MASK_ID)

    # ---- print: each strategy split into commit | still-mask region ----
    print("-" * 118)
    print(f"  ({tag}) block {bs}:{be}  masked={int(masked.sum())}")
    print(f"  think_1   COMMIT[{int(cm1.sum())}]: {dtxt(am1[0], cm1[0])!r}")
    print(f"  think_1   STILL [{int(sm1.sum())}]: {dtxt(a1[0],  sm1[0])!r}")
    print(f"  talk_1    COMMIT[{int(cmT.sum())}]: {dtxt(amT[0], cmT[0])!r}")
    print(f"  talk_1    STILL [{int(smT.sum())}]: {dtxt(aT[0],  smT[0])!r}")
    print(f"  think_2   COMMITS +{int(cm2.sum())}: {dtxt(am2[0], cm2[0])!r}   (think_1 left {int(sm1.sum())} masked)")
    print(f"  think_2   STILL [{int(sm2.sum())}]: {dtxt(a2[0], sm2[0])!r}")
    print(f"  talk_think_converged: COMMIT[{int(ttk_commit.sum())}] STILL[{int(ttk_still.sum())}]  "
          f"FWD = 1 think + 1 talk + {ttk_iters} think = {ttk_fwd} total | wall = {1e3 * ttk_wall:.0f} ms")
    print(f"      committed: {dtxt(ttk_ids[0], ttk_commit[0])!r}")
    print(f"  base_think  CONVERGED: COMMIT[{int((masked & (base_ids != MASK_ID)).sum())}] STILL[{int(base_still.sum())}]  "
          f"FWD = {base_fwd} think | wall = {1e3 * t_base:.0f} ms")
    print(f"      committed: {dtxt(base_ids[0], masked[0])!r}")
    print(f"  think+think+talk... CONVERGED: COMMIT[{int(c_commit.sum())}] STILL[{int(c_still.sum())}]  "
          f"FWD = {c_nth} think + {c_ntk} talk = {c_nth + c_ntk} total | wall = {1e3 * t_c:.0f} ms")
    print(f"      committed: {dtxt(c_ids[0], c_commit[0])!r}")
    print(f"  think+talk+think+talk... CONVERGED: COMMIT[{int(d_commit.sum())}] STILL[{int(d_still.sum())}]  "
          f"FWD = {d_nth} think + {d_ntk} talk = {d_nth + d_ntk} total | wall = {1e3 * t_d:.0f} ms")
    print(f"      committed: {dtxt(d_ids[0], d_commit[0])!r}")

    # ---- per-position table over the masked block ----
    def cell(a, c, j, cm=None, w=12):
        t = repr(tok.decode([int(a[0, j])]))[:w]
        flag = "" if cm is None else (" C" if bool(cm[0, j]) else " M")
        return f"{t:>{w}}({c[0, j]:.2f}){flag}"
    print(f"  {'pos':>3} | {'think_1 C/M':>17} | {'talk_1 C/M':>17} | {'think_2 C/M':>17} | {'ttk_conv':>15} | {'base_think':>15}")
    for j in range(B):
        if not bool(masked[0, j]):
            continue
        print(f"  {bs+j:>3} | {cell(a1,c1,j,cm1):>17} | {cell(aT,cT,j,cmT):>17} | {cell(a2,c2,j,(cm1|cm2)):>17} | "
              f"{cell(atk,ctk,j,w=9):>15} | {cell(ab,cb,j,w=9):>15}")

    # ---- aggregates ----
    for name, reg in (("commit", cm1[0]), ("still", sm1[0])):
        if int(reg.sum()):
            agg[name][0] += int((aT[0][reg] == a2[0][reg]).sum())
            agg[name][1] += int(reg.sum())
    agg["ttk_vs_base"][0] += int((atk[0][masked[0]] == ab[0][masked[0]]).sum())
    agg["ttk_vs_base"][1] += int(masked.sum())
    for key, val in (("ttk_fwd", ttk_fwd), ("ttk_wall", ttk_wall), ("base_fwd", base_fwd), ("base_wall", t_base),
                     ("c_fwd", c_nth + c_ntk), ("c_wall", t_c), ("d_fwd", d_nth + d_ntk), ("d_wall", t_d)):
        agg[key][0] += val
        agg[key][1] += 1
    agg["c_vs_base"][0] += int((ac[0][masked[0]] == ab[0][masked[0]]).sum())
    agg["c_vs_base"][1] += int(masked.sum())
    agg["d_vs_base"][0] += int((ad[0][masked[0]] == ab[0][masked[0]]).sum())
    agg["d_vs_base"][1] += int(masked.sum())
    cr, sr = cm1[0], sm1[0]
    ac = (aT[0][cr] == a2[0][cr]).float().mean().item() if int(cr.sum()) else float("nan")
    as_ = (aT[0][sr] == a2[0][sr]).float().mean().item() if int(sr.sum()) else float("nan")
    tb = (atk[0][masked[0]] == ab[0][masked[0]]).float().mean().item()
    print(f"  -> talk_1 vs think_2:  COMMIT={ac:.2f}  STILL={as_:.2f}    |    talk_think_converged vs base_think: {tb:.2f}")
    return base_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--think_path", required=True)
    ap.add_argument("--talk_path", required=True, help="trained talk ckpt (or ../merged_10L untrained)")
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--max_iters", type=int, default=32, help="cap for 'until converged' think decode (>= block_length)")
    ap.add_argument("--gen_block", type=int, default=1,
                    help="first GENERATION block to probe (>=1 => fully masked, conditioned on think-decoded "
                         "earlier blocks). The probe does THIS block and the NEXT one.")
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

    agg = {"commit": [0, 0], "still": [0, 0], "ttk_vs_base": [0, 0],
           "ttk_fwd": [0, 0], "ttk_wall": [0, 0], "base_fwd": [0, 0], "base_wall": [0, 0],
           "c_fwd": [0, 0], "c_wall": [0, 0], "c_vs_base": [0, 0],
           "d_fwd": [0, 0], "d_wall": [0, 0], "d_vs_base": [0, 0]}
    rows = list(load_dataset("gsm8k", "main", split="test"))[:args.limit]
    for i, row in enumerate(rows):
        messages = [{"role": "user", "content": GSM8K_USER_TEMPLATE.format(question=row["question"])}]
        prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=True, return_tensors="pt").to(args.device)
        P = prompt_ids.shape[1]
        first_b = P // B
        target_b = first_b + args.gen_block
        L = (target_b + 2) * B                                  # room for the two probed blocks
        x = torch.full((1, L), MASK_ID, dtype=torch.long, device=args.device)
        x[:, :P] = prompt_ids
        attn = build_block_causal_mask(L, B, dtype, args.device)
        pos = torch.arange(L, device=args.device).unsqueeze(0)
        # decode the blocks BEFORE the first probed block with think (clean preceding context)
        for b in range(first_b, target_b):
            be0 = (b + 1) * B
            think_converge(think, emb, x, b * B, be0, attn[:, :, :be0, :be0], pos[:, :be0], args.threshold, args.max_iters)

        print("=" * 118)
        print(f"[{i}] Q: {row['question'][:96]}…")
        # block A (this) and block B (next). block B's preceding context = base_think's decode of block A.
        for k, blk in enumerate((target_b, target_b + 1)):
            bs, be = blk * B, (blk + 1) * B
            m, p = attn[:, :, :be, :be], pos[:, :be]
            base_ids = probe_block(think, talk, emb, tok, x, bs, be, B, m, p, args, agg,
                                   tag=f"block {k + 1}/2")
            x[:, bs:be] = base_ids                              # commit base_think's decode -> context for next block

    print("=" * 118)
    print(f"[TOTAL] talk_1 vs think_2  [commit]: {agg['commit'][0]}/{agg['commit'][1]} = {agg['commit'][0]/max(1,agg['commit'][1]):.3f}")
    print(f"[TOTAL] talk_1 vs think_2  [still ]: {agg['still'][0]}/{agg['still'][1]} = {agg['still'][0]/max(1,agg['still'][1]):.3f}")
    print(f"[TOTAL] talk_think_converged vs base_think (does think recover the talk's commit?): "
          f"{agg['ttk_vs_base'][0]}/{agg['ttk_vs_base'][1]} = {agg['ttk_vs_base'][0]/max(1,agg['ttk_vs_base'][1]):.3f}")
    nb = max(1, agg["base_fwd"][1])
    print(f"[TOTAL] vs base_think agreement:  think+think+talk = {agg['c_vs_base'][0]/max(1,agg['c_vs_base'][1]):.3f}  "
          f"think+talk+talk... (alt) = {agg['d_vs_base'][0]/max(1,agg['d_vs_base'][1]):.3f}")
    print(f"[TOTAL] avg FORWARDS/block (think+talk):")
    print(f"          base_think           = {agg['base_fwd'][0]/nb:.1f}")
    print(f"          talk_think_converged = {agg['ttk_fwd'][0]/nb:.1f}")
    print(f"          think+think+talk...  = {agg['c_fwd'][0]/nb:.1f}")
    print(f"          think+talk+think...  = {agg['d_fwd'][0]/nb:.1f}")
    print(f"[TOTAL] avg WALL/block:  base_think = {1e3*agg['base_wall'][0]/nb:.0f} ms   "
          f"talk_think = {1e3*agg['ttk_wall'][0]/nb:.0f} ms   "
          f"think+think+talk = {1e3*agg['c_wall'][0]/nb:.0f} ms   "
          f"think+talk+talk(alt) = {1e3*agg['d_wall'][0]/nb:.0f} ms")
    print("Read: a mixed method WINS if it matches base_think's answer (high vs-base agreement) at LOWER wall "
          "than base_think. Talk forwards are ~half a think's FLOPs, so wall is the honest cost; if a method "
          "needs more total forwards but more of them are cheap talk passes, it can still be faster.")


if __name__ == "__main__":
    main()
