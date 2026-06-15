"""Toy probe: compare, on the FIRST generation block of a few GSM8K prompts (block fully masked),
the per-position predictions of four things, each split into the COMMIT region (the confident
left-to-right prefix that decode_uniform would commit at tau) vs the STILL-MASK region (the
deferred tail, where the \\n-filler collapse lives):

  think_1        = think(X) ONCE                                  (think's 1st pass)
  talk_1         = talk(X + think_1 top-K soft-embed) ONCE        (1 think + 1 talk = a lightweight 2nd
                                                                   forward; its reference is think_2)
  think_2        = think_1 -> commit think_1's prefix -> think    (think's REAL 2nd iteration)
  talk_think_k   = think_1 -> talk_1 -> commit talk_1's prefix    (does think CONVERGE well when seeded
                   -> think x k                                    from the TALK's commit? = the Stage-2
                                                                   "teacher from the talk's state" question)

The decisive reads:
  * COMMIT region: think_1 / talk_1 / think_2 should largely AGREE (easy confident tokens).
  * STILL-MASK region: does think_1 flood high-confidence \\n/filler? does talk_1 mimic that filler or
    does it match think_2 (resolve content)? -> talk_1-vs-think_2 agreement IN the still-mask region is
    the number that matters (can the lightweight 2nd forward leap one think iteration on the HARD tail?).
  * talk_think_k vs think_2/think_1: if think converges fine from the talk's commit, the talk's commits
    are good enough to seed think (supports the Stage-2 think-from-talk-state teacher).

Run (on the GPU box):
  python -m tasks.t3d_probe_converged_teacher \
      --think_path ../DMax-Math-16B-moe-merge \
      --talk_path  ./t3_topk_stage2_onpolicy_outputs/checkpoints/<step>/hf_ckpt \
      --block_length 32 --top_k 10 --threshold 0.3 --k_iters 4 --limit 5
"""

from __future__ import annotations

import argparse
import torch

from tasks.t3d_topk_talk import load_causal_lm, confident_prefix_commit
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
    ap.add_argument("--k_iters", type=int, default=4, help="trailing think iterations for talk_think_k")
    ap.add_argument("--gen_block", type=int, default=1,
                    help="which GENERATION block to probe. 0 = the first block (mostly PROMPT -> few masked "
                         "positions). >=1 = a FULLY-masked block (all 32 = response), conditioned on the "
                         "earlier blocks decoded by think first. Use 1 to see the still-mask filler tail.")
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

    def dtxt(ids_row, region_row):                              # decode the tokens of a [B] id row at a [B] bool region
        sel = ids_row[region_row]
        return tok.decode(sel, skip_special_tokens=False) if sel.numel() else "(empty)"

    # aggregates: talk_1 vs think_2 agreement, split by think_1's commit vs still-mask region
    agg = {"commit": [0, 0], "still": [0, 0], "ttk_vs_t2": [0, 0]}
    rows = list(load_dataset("gsm8k", "main", split="test"))[:args.limit]
    for i, row in enumerate(rows):
        messages = [{"role": "user", "content": GSM8K_USER_TEMPLATE.format(question=row["question"])}]
        prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=True, return_tensors="pt").to(args.device)
        P = prompt_ids.shape[1]
        first_b = P // B                                        # block containing the prompt's end
        target_b = first_b + args.gen_block                     # the block we probe
        bs, be = target_b * B, (target_b + 1) * B
        L = be
        x = torch.full((1, L), MASK_ID, dtype=torch.long, device=args.device)
        x[:, :P] = prompt_ids
        attn = build_block_causal_mask(L, B, dtype, args.device)
        pos = torch.arange(L, device=args.device).unsqueeze(0)
        # decode the PRECEDING blocks with think (realistic committed context) so the probed block is fresh
        for b in range(first_b, target_b):
            b_end = (b + 1) * B
            think_k_iters(think, emb, x, b * B, b_end, attn[:, :, :b_end, :b_end], pos[:, :b_end],
                          args.threshold, args.k_iters + 6)
        m, p = attn[:, :, :be, :be], pos[:, :be]
        masked = (x[:, bs:be] == MASK_ID)                       # [1, B] masked positions in the probed block

        # ---- think_1 (one pass) + its commit region ----
        th1 = think(inputs_embeds=emb(x[:, :be]), attention_mask=m, position_ids=p,
                    use_cache=False, return_dict=True).logits[:, bs:be]
        a1, c1 = _conf(th1)
        am1, cm1 = confident_prefix_commit(th1, masked, B, args.threshold)   # think_1 commit prefix
        sm1 = masked & (~cm1)                                                # think_1 still-mask region

        # ---- talk_1 (one think + one talk) + its commit region ----
        soft = build_topk_soft_embeds(th1, emb, MASK_ID, top_k=args.top_k,
                                      keep_mask_residual=not args.no_mask_residual)
        inp = emb(x[:, :be]).clone()
        inp[:, bs:be][masked] = soft[masked].to(inp.dtype)
        ta1 = talk(inputs_embeds=inp, attention_mask=m, position_ids=p,
                   use_cache=False, return_dict=True).logits[:, bs:be]
        aT, cT = _conf(ta1)
        amT, cmT = confident_prefix_commit(ta1, masked, B, args.threshold)   # talk_1 commit prefix
        smT = masked & (~cmT)

        # ---- think_2 = think_1 -> commit think_1's prefix (hard) -> think ----
        x1 = x.clone()
        x1[:, bs:be] = torch.where(cm1, am1, x1[:, bs:be])
        th2 = think(inputs_embeds=emb(x1[:, :be]), attention_mask=m, position_ids=p,
                    use_cache=False, return_dict=True).logits[:, bs:be]
        a2, c2 = _conf(th2)

        # ---- talk_think_k = think_1 -> talk_1 -> commit talk_1's prefix (hard) -> think x k ----
        xt = x.clone()
        xt[:, bs:be] = torch.where(cmT, amT, xt[:, bs:be])
        ttk_logits, ttk_ids = think_k_iters(think, emb, xt, bs, be, m, p, args.threshold, args.k_iters)
        atk, ctk = _conf(ttk_logits)

        # ---- print: each output split into commit | still-mask region ----
        print("=" * 110)
        print(f"[{i}] Q: {row['question'][:88]}…   (block {bs}:{be}, masked={int(masked.sum())})")
        print(f"  think_1  COMMIT[{int(cm1.sum())}]: {dtxt(am1[0], cm1[0])!r}")
        print(f"  think_1  STILL [{int(sm1.sum())}]: {dtxt(a1[0],  sm1[0])!r}")
        print(f"  talk_1   COMMIT[{int(cmT.sum())}]: {dtxt(amT[0], cmT[0])!r}")
        print(f"  talk_1   STILL [{int(smT.sum())}]: {dtxt(aT[0],  smT[0])!r}")
        print(f"  think_2  (think+think) @still_1[{int(sm1.sum())}]: {dtxt(a2[0], sm1[0])!r}")
        print(f"  talk_think_k (1think+1talk+{args.k_iters}think) committed block: {dtxt(ttk_ids[0], masked[0])!r}")

        # ---- per-position table over the originally-masked block ----
        print(f"  {'pos':>3} | {'think_1(conf)C/M':>20} | {'talk_1(conf)C/M':>20} | {'think_2(conf)':>16} | {'ttk(conf)':>16}")
        for j in range(B):
            if not bool(masked[0, j]):
                continue
            def cell(a, c, jj, cm=None):
                t = repr(tok.decode([int(a[0, jj])]))[:12]
                flag = "" if cm is None else (" C" if bool(cm[0, jj]) else " M")
                return f"{t:>12}({c[0, jj]:.2f}){flag}"
            print(f"  {bs+j:>3} | {cell(a1,c1,j,cm1):>20} | {cell(aT,cT,j,cmT):>20} | "
                  f"{cell(a2,c2,j):>16} | {cell(atk,ctk,j):>16}")

        # ---- aggregate: talk_1 vs think_2 (the lightweight-2nd-forward reference), split by region ----
        for name, reg in (("commit", cm1[0]), ("still", sm1[0])):
            if int(reg.sum()):
                agg[name][0] += int((aT[0][reg] == a2[0][reg]).sum())
                agg[name][1] += int(reg.sum())
        agg["ttk_vs_t2"][0] += int((atk[0][masked[0]] == a2[0][masked[0]]).sum())
        agg["ttk_vs_t2"][1] += int(masked.sum())
        cr, sr = cm1[0], sm1[0]
        ac = (aT[0][cr] == a2[0][cr]).float().mean().item() if int(cr.sum()) else float("nan")
        as_ = (aT[0][sr] == a2[0][sr]).float().mean().item() if int(sr.sum()) else float("nan")
        print(f"  -> talk_1 vs think_2 agreement:  COMMIT region={ac:.2f}  STILL-MASK region={as_:.2f}")

    print("=" * 110)
    for k, (n, d) in agg.items():
        if k == "ttk_vs_t2":
            print(f"[TOTAL] talk_think_k vs think_2 (does think converge from the talk's commit?): {n}/{d} = {n/max(1,d):.3f}")
        else:
            print(f"[TOTAL] talk_1 vs think_2  [{k} region]: {n}/{d} = {n/max(1,d):.3f}")
    print("Read: high agreement in COMMIT, LOW in STILL-MASK => the lightweight 2nd forward leaps the easy "
          "tokens but NOT the hard deferred tail (where think_1 floods filler). That tail is where the gold "
          "CE must carry it (think_2/think's distribution there is unreliable as a KL teacher).")


if __name__ == "__main__":
    main()
