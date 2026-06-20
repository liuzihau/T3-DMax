"""T3-D top-K talk: GSM8K decode + eval.

Three decode methods (--decode_mode), all in the SAME unified decoder (decode_t3d). A per-block
schedule(i)->'think'|'talk' picks the model for step i; whichever model runs COMMITS its own confident
left-to-right prefix (the single DMax threshold rule, + EOS seq-stop). The methods are just schedules:

  * base_think        : think only (pure DMax baseline).
  * think_then_talk a : think x a (commit), then talk takes over until the block converges.
                        (a = --think_passes)
  * interleave t      : repeating (think x t, then 1 talk). t=3 -> think think think talk think... .
                        (t = --think_per_talk; t=1 is strict think<->talk alternation)

Unified per-forward input (no flags -- this is always how inference feeds the model):
  * COMMITTED positions: the DMax soft top-K(+mask-residual) blend of the latest logits, renormalized
    to the embedding manifold (decode_uniform's soft_cond) -- never hard tokens.
  * MASKED positions: bare [MASK] when the forward model is THINK (think generates); the top-K soft
    blend (+mask residual) of the previous pass's logits when the model is TALK.

Compute: think_forwards (20-layer) + talk_forwards (10-layer). 20-layer-equivalent cost =
think_fwd + 0.5*talk_fwd, vs full DMax = iters*1.0.

Run:
  python -m tasks.t3d_topk_eval_gsm8k \
    --think_path ../DMax-Math-16B-moe-merge \
    --talk_path  ./t3_topk_stage2_onpolicy_outputs/checkpoints/<step>/hf_ckpt  (or ../merged_10L) \
    --decode_mode think_then_talk --think_passes 1 --gen_length 512 --block_length 32 --threshold 0.3 --limit 200
"""

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from tasks.t3d_topk_talk import load_causal_lm
from tasks.t3d_topk_soft_embed import build_block_input

GSM8K_USER_TEMPLATE = "Question: {question}\nLet's think step by step\nAnswer:"


# ---- rollout-convergence trace (diagnostics: confidence / commits / fresh-candidate overlap) --------
class RolloutTrace:
    """Accumulates per-pass / per-block / per-sequence stats of a decode run. Signatures:
      * commit STARVATION  -> commits/pass ~1 (only the first-mask fallback), passes-to-converge high.
      * coverage DRIFT     -> overlap-with-iter0 falls across pass index (commits leave iter-0's pool).
      * soft-embed DEGENERACY/repetition -> high adjacent-repeat fraction in committed blocks.
    overlap_iter0 = of the tokens COMMITTED this pass, the fraction that lie in the FIRST pass's (iter-0)
    top-K (1.0 at pass 0; a fall = fresh tokens entering the pool, e.g. think re-forwarding with context).
    overlap_lastT = same fraction but against the PREVIOUS think pass's top-K (a warm, just-recomputed ref).
    iter0 falling while lastT stays high = pure STALENESS (each fresh think agrees with the last one; a
    refresh recovers). BOTH falling = genuine churn (commits leave even the latest think's pool)."""

    def __init__(self):
        self.p_commit = defaultdict(float); self.p_conf = defaultdict(float)
        self.p_ovl = defaultdict(float); self.p_ovl_n = defaultdict(int); self.p_n = defaultdict(int)
        self.p_ovl2 = defaultdict(float); self.p_ovl2_n = defaultdict(int)
        self.block_passes = []; self.block_rep = []; self.n_blocks = 0; self.n_capped = 0
        self.n_seq = 0; self.n_no_eos = 0

    def add_pass(self, it, n_commit, mean_conf, ovl, ovl2=float("nan")):
        self.p_commit[it] += n_commit; self.p_conf[it] += mean_conf; self.p_n[it] += 1
        if ovl == ovl:                                              # not NaN (commits happened)
            self.p_ovl[it] += ovl; self.p_ovl_n[it] += 1
        if ovl2 == ovl2:                                            # not NaN (a prior think pass existed)
            self.p_ovl2[it] += ovl2; self.p_ovl2_n[it] += 1

    def add_block(self, passes, capped, rep):
        self.block_passes.append(passes); self.block_rep.append(rep)
        self.n_blocks += 1; self.n_capped += int(capped)

    def add_seq(self, no_eos):
        self.n_seq += 1; self.n_no_eos += int(no_eos)

    def report(self):
        import statistics as st
        print("=" * 78)
        print("[trace] rollout convergence (per within-block pass index)")
        print(f"  {'pass':>4} | {'n_blk':>5} | {'commits':>7} | {'conf':>9} | {'overlap_iter0':>15} | {'overlap_lastT':>15}")
        for it in sorted(self.p_n):
            n = self.p_n[it]
            ovl = self.p_ovl[it] / self.p_ovl_n[it] if self.p_ovl_n[it] else float("nan")
            ovl2 = self.p_ovl2[it] / self.p_ovl2_n[it] if self.p_ovl2_n[it] else float("nan")
            print(f"  {it:>4} | {n:>5} | {self.p_commit[it]/n:>7.2f} | {self.p_conf[it]/n:>9.3f} | {ovl:>15.3f} | {ovl2:>15.3f}")
        bp = self.block_passes or [0]
        print("-" * 78)
        print(f"[trace] blocks={self.n_blocks}  passes/block: mean={st.mean(bp):.1f} max={max(bp)}  "
              f"capped(hit max_iters)={self.n_capped}/{self.n_blocks} ({self.n_capped/max(1,self.n_blocks):.1%})")
        print(f"[trace] adjacent-repeat frac/block: mean={st.mean(self.block_rep or [0]):.3f}  "
              f"no-EOS seqs={self.n_no_eos}/{self.n_seq} ({self.n_no_eos/max(1,self.n_seq):.1%})")
        its = sorted(self.p_n)
        early = [i for i in its if i < max(1, len(its)//3)]; late = [i for i in its if i >= 2*len(its)//3]
        def _m(d, keys, dn=None):
            num = sum(d[i] for i in keys); den = sum((dn or self.p_n)[i] for i in keys)
            return num/den if den else float("nan")
        ovl_e = _m(self.p_ovl, early, self.p_ovl_n); ovl_l = _m(self.p_ovl, late, self.p_ovl_n)
        ovl2_e = _m(self.p_ovl2, early, self.p_ovl2_n); ovl2_l = _m(self.p_ovl2, late, self.p_ovl2_n)
        commits_l = _m(self.p_commit, late)
        rep = st.mean(self.block_rep or [0])
        print("-" * 78)
        print(f"[trace] signals: overlap iter0 early={ovl_e:.3f} -> late={ovl_l:.3f} (STALENESS if falling); "
              f"overlap lastT early={ovl2_e:.3f} -> late={ovl2_l:.3f} (CHURN if also falling, else refresh recovers); "
              f"late commits/pass={commits_l:.2f} (STARVATION if ~1); rep={rep:.3f} (DEGENERACY if high)")


# ---- decode helpers (the single commit rule + block-causal mask) ----------------
def build_block_causal_mask(L, block_length, dtype, device):
    idx = torch.arange(L, device=device)
    q_block = (idx // block_length).unsqueeze(1)
    kv_block = (idx // block_length).unsqueeze(0)
    allowed = (kv_block <= q_block)
    mask = torch.zeros(1, 1, L, L, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


def dmax_commit_uniform(logits, mask_index, active_index, threshold):
    """DMax's left-to-right high-confidence prefix commit + 0.9 Breakflag. Commits the leftmost
    contiguous run of masked positions whose argmax confidence >= threshold; if none clears the bar,
    still commits the first masked position (guarantees >=1 commit/pass so a block always finishes).
    The single commit rule for all decode methods (the only inference commit mechanism)."""
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


# ---- the 3 decode methods (a per-block schedule; whichever model runs commits) -----------------
def _schedule(mode, think_passes, think_per_talk):
    """Return schedule(i)->'think'|'talk' for the 3 decode methods."""
    if mode == "base_think":                       # method 1: pure think (DMax baseline)
        return lambda i: "think"
    if mode == "think_then_talk":                  # method 2: think x a (commit), then talk to converge
        a = max(1, think_passes)
        return lambda i: "think" if i < a else "talk"
    if mode == "interleave":                       # method 3: repeating (think x t, then 1 talk)
        t = max(1, think_per_talk)
        return lambda i: "think" if (i % (t + 1)) < t else "talk"
    raise ValueError(f"unknown decode mode: {mode}")


@torch.no_grad()
def decode_t3d(think, talk, emb, mask_id, prompt_ids, *, schedule, gen_length, block_length,
               threshold, top_k, max_iters, device, dtype, early_stop=False, eos_id=None, trace=None):
    """Unified T3-D inference. Per block, a per-step schedule picks think|talk; that model commits its
    own confident left-to-right prefix (dmax_commit_uniform). Input each forward: COMMITTED positions
    always get the DMax soft top-K(+mask-residual) blend of the latest logits; MASKED positions get
    bare [MASK] (think) or the top-K soft blend of the previous pass's logits (talk).

    Block-end: by default loop until the block is fully committed (no masked left), capped at max_iters
    (=block_length; the prefix rule commits >=1/step so it converges in <=block_length). early_stop=True
    also ends a block on the DMax Breakflag (all active >=0.9 or no-change) and stops generating further
    blocks once a block commits EOS (batch-filtering is a no-op at batch=1).
    Returns (response_ids, think_fwd, talk_fwd)."""
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

        def block_input(src_logits, model):                        # the single shared input rule
            return build_block_input(x, bs, be, emb, src_logits, model, mask_id, top_k)

        cand = None                                                # the previous pass's block logits
        block_topk = None                                          # iter-0 top-K ids [1,B,K] (overlap ref)
        last_think_topk = None                                     # previous think pass's top-K ids [1,B,K]
        passes = 0
        for it in range(max_iters):
            block_x = x[:, bs:be]
            mask_index = (block_x == mask_id)
            if not bool(mask_index.any()):
                break
            if schedule(it) == "think":                            # THINK commits: [MASK] at masked, soft committed
                logits = think(inputs_embeds=block_input(cand, "think"), attention_mask=m,
                               position_ids=p, use_cache=False, return_dict=True).logits[:, bs:be]
                think_fwd += 1
            else:                                                  # TALK commits: top-K soft at masked
                if cand is None:                                   # block starts on talk -> 1 bootstrap think
                    cand = think(inputs_embeds=block_input(None, "think"), attention_mask=m, position_ids=p,
                                 use_cache=False, return_dict=True).logits[:, bs:be]
                    think_fwd += 1
                logits = talk(inputs_embeds=block_input(cand, "talk"), attention_mask=m,
                              position_ids=p, use_cache=False, return_dict=True).logits[:, bs:be]
                talk_fwd += 1
            cand = logits                                          # feed this pass's top-K forward
            x0, high_conf, max_probs, breakflag = dmax_commit_uniform(logits, mask_index, mask_index, threshold)
            committed = high_conf & mask_index
            new_block = torch.where(high_conf, x0, block_x)
            changed = bool((new_block != block_x).any())
            x[:, bs:be] = new_block
            passes += 1
            if trace is not None:
                if block_topk is None:                             # snapshot iter-0 top-K as the overlap ref
                    block_topk = logits.topk(top_k, dim=-1).indices
                n_commit = int(committed.sum())
                mean_conf = float(max_probs[mask_index].mean()) if bool(mask_index.any()) else float("nan")
                if n_commit:
                    in_topk = (block_topk[0] == x0[0].unsqueeze(-1)).any(-1)       # [B] bool, vs iter-0
                    ovl = float(in_topk[committed[0]].float().mean())
                    if last_think_topk is not None:                # vs the PREVIOUS think pass's top-K
                        in_topk2 = (last_think_topk[0] == x0[0].unsqueeze(-1)).any(-1)
                        ovl2 = float(in_topk2[committed[0]].float().mean())
                    else:
                        ovl2 = float("nan")
                else:
                    ovl = ovl2 = float("nan")
                trace.add_pass(it, n_commit, mean_conf, ovl, ovl2)
                if schedule(it) == "think":                        # refresh the rolling think ref AFTER scoring
                    last_think_topk = logits.topk(top_k, dim=-1).indices
            if early_stop and (breakflag or not changed):
                break
        if trace is not None:
            blk = x[:, bs:be]
            rep = float((blk[0, 1:] == blk[0, :-1]).float().mean()) if block_length > 1 else 0.0
            trace.add_block(passes, capped=bool((blk == mask_id).any()) or passes >= max_iters, rep=rep)
        if early_stop and eos_id is not None and bool((x[:, bs:be] == eos_id).any()):
            if be < L:                                             # EOS in this block -> rest of seq is done
                x[:, be:] = eos_id
            break
    if trace is not None:
        trace.add_seq(no_eos=(eos_id is not None and not bool((x[:, P:P + gen_length] == eos_id).any())))
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
    ap.add_argument("--decode_mode", default="think_then_talk",
                    choices=["base_think", "think_then_talk", "interleave"],
                    help="base_think = think only (DMax baseline); think_then_talk = think x a (commit) then "
                         "talk (a=--think_passes); interleave = repeating (think x t, then 1 talk) "
                         "(t=--think_per_talk).")
    ap.add_argument("--think_passes", type=int, default=1,
                    help="[think_then_talk] number of leading think COMMIT passes before talk takes over.")
    ap.add_argument("--think_per_talk", type=int, default=3,
                    help="[interleave] think COMMIT passes between each talk pass (t=1 = strict alternation).")
    ap.add_argument("--gen_length", type=int, default=512)
    ap.add_argument("--block_length", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--max_iters", type=int, default=32)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--debug_print", action="store_true")
    ap.add_argument("--early_stop", action="store_true",
                    help="DMax termination: per-block end on all-active>=0.9 OR no-change (decode_uniform "
                         "Breakflag), AND stop generating once a block commits EOS. Off = run each block to "
                         "full convergence (cap --max_iters).")
    ap.add_argument("--eos_id", type=int, default=156892, help="EOS token id (DMax LLaDA-2.0 = 156892).")
    ap.add_argument("--trace", action="store_true",
                    help="collect a rollout-convergence trace (per-pass commits / confidence / overlap-with-"
                         "iter0-topK, per-block passes+repetition, no-EOS). Use on base_think for the think "
                         "baseline, or on think_then_talk/interleave to see talk's confidence + fresh candidates.")
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
    sched = _schedule(args.decode_mode, args.think_passes, args.think_per_talk)
    trace = RolloutTrace() if args.trace else None
    print(f"[eval] {len(rows)} problems  mode={args.decode_mode} gen={args.gen_length} "
          f"block={args.block_length} top_k={args.top_k}")

    n_ok = tot_think = tot_talk = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        messages = [{"role": "user", "content": GSM8K_USER_TEMPLATE.format(question=row["question"])}]
        prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=True, return_tensors="pt").to(args.device)
        resp, th, tk = decode_t3d(think, talk, emb, mask_id, prompt_ids, schedule=sched,
                                  gen_length=args.gen_length, block_length=args.block_length,
                                  threshold=args.threshold, top_k=args.top_k,
                                  max_iters=args.max_iters, device=args.device, dtype=dtype,
                                  early_stop=args.early_stop, eos_id=args.eos_id, trace=trace)
        text = tok.decode(resp[0], skip_special_tokens=True)
        gold, pred = gold_answer(row["answer"]), pred_answer(text)
        ok = is_correct(pred, gold)
        n_ok += ok; tot_think += th; tot_talk += tk
        if args.debug_print or (i < 3):
            print(f"[{i}] {'OK ' if ok else 'XX '} pred={pred} gold={gold} think={th} talk={tk} tail={text[-120:]!r}")
        if args.debug_print:                                   # FULL generation (not just the tail)
            print(f"  ---- [{i}] question ----\n{row['question']}")
            print(f"  ---- [{i}] FULL generated ({len(text)} chars) ----\n{text}")
            print(f"  ---- [{i}] end (pred={pred} gold={gold}) ----")
        if (i + 1) % 25 == 0:
            print(f"  …{i+1}/{len(rows)}  acc={n_ok/(i+1):.3f}")

    n = len(rows)
    cost20 = tot_think + 0.5 * tot_talk            # 20-layer-equivalent forwards
    print("=" * 70)
    print(f"[eval] mode={args.decode_mode}  GSM8K acc = {n_ok}/{n} = {n_ok/n:.3f}")
    print(f"[eval] mean think/ex={tot_think/n:.1f}  talk/ex={tot_talk/n:.1f}  "
          f"20L-equiv/ex={cost20/n:.1f}  (full DMax ≈ iters*1.0)")
    print(f"[eval] {time.time()-t0:.0f}s. Baseline to beat: 84% @ gen512.")
    if trace is not None:
        trace.report()


if __name__ == "__main__":
    main()
