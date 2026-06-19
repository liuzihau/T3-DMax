"""Block 4 — anchor-free top-K talk: GSM8K decode + eval.

Scores a trained top-K talk against the full-DMax baseline over the WHOLE GSM8K
response. Five decode methods are available via --decode_mode (see DECODE_MODES
below). They fall into two families:

  (A) think-as-CANDIDATE  [think never commits, TALK always commits]
      * seed  : per block, THINK runs once (or --seed_passes times) producing
                per-position top-K candidates; TALK iterates -- still-masked
                positions fed think's (then its OWN) top-K soft-embedding,
                committed positions keep their token embedding. Commit at the
                DMax decode_uniform rule (--threshold), 0.9 early-stop. This is
                the H1 inference dynamic the training targets (think once/block).

  (B) think-as-DECODER  [whichever model runs that step COMMITS its own prefix]
      ported from t3d_probe_converged_teacher.mixed_converge. Per block a
      schedule(i)->'think'|'talk' picks the model for step i; a think step feeds
      bare [MASK] at undecided positions, a talk step feeds the top-K soft-embed
      of the PREVIOUS pass's logits (think's if the prev step was think, else the
      talk's own). Loops UNTIL the block is fully committed, capped at --max_iters
      (=block_length=32) decoding steps. Modes:
      * cross           : think->talk->think->talk... (strict alternation)
      * think_then_talk : think x N (commit) then talk... (N=--think_seed_count, default 2)
      * cycle           : REPEATING (think x A, talk x B); A=--think_per_cycle, B=--talk_per_cycle
                          (e.g. 2,1 = think,think,talk,think,think,talk... -- a think-heavy cross)
      * think_only      : pure DMax baseline (think decoded alone to convergence)

Compute: think_forwards (20-layer) + talk_forwards (10-layer). The 20-layer-
equivalent cost = think_fwd + 0.5*talk_fwd, vs full DMax = iters*1.0.

Run:
  python -m tasks.t3d_topk_eval_gsm8k \
    --think_path ../DMax-Math-16B-moe-merge \
    --talk_path  ./t3_topk_talk_merged_only_outputs/<hf_ckpt>  (or ../merged_10L untrained baseline) \
    --decode_mode seed --gen_length 512 --block_length 32 --threshold 0.3 --top_k 10 --limit 200
"""

from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from tasks.t3d_topk_talk import load_causal_lm
from tasks.t3d_topk_soft_embed import build_topk_soft_embeds

GSM8K_USER_TEMPLATE = "Question: {question}\nLet's think step by step\nAnswer:"


# ---- seed-rollout convergence trace (diagnoses the talk self-rollout collapse) -----------------
class RolloutTrace:
    """Accumulates per-pass / per-block / per-sequence stats of the `seed` decode to pin the collapse
    mode. The three suspects and their signatures:
      * commit STARVATION  -> commits/pass ~1 (only the first-mask fallback), passes-to-converge high.
      * coverage DRIFT     -> overlap-with-think-seed falls across pass index (talk leaves think's support).
      * soft-embed DEGENERACY/repetition -> high adjacent-repeat fraction in committed blocks.
    overlap = of the tokens talk COMMITS this pass, the fraction that lie in THINK's original seed top-K
    at that position (1.0 at the seed pass; a fall over later passes = drift)."""

    def __init__(self):
        self.p_commit = defaultdict(float); self.p_conf = defaultdict(float)
        self.p_ovl = defaultdict(float); self.p_ovl_n = defaultdict(int); self.p_n = defaultdict(int)
        self.block_passes = []; self.block_rep = []; self.n_blocks = 0; self.n_capped = 0
        self.n_seq = 0; self.n_no_eos = 0

    def add_pass(self, it, n_commit, mean_conf, ovl):
        self.p_commit[it] += n_commit; self.p_conf[it] += mean_conf; self.p_n[it] += 1
        if ovl == ovl:                                              # not NaN (commits happened)
            self.p_ovl[it] += ovl; self.p_ovl_n[it] += 1

    def add_block(self, passes, capped, rep):
        self.block_passes.append(passes); self.block_rep.append(rep)
        self.n_blocks += 1; self.n_capped += int(capped)

    def add_seq(self, no_eos):
        self.n_seq += 1; self.n_no_eos += int(no_eos)

    def report(self):
        import statistics as st
        print("=" * 78)
        print("[trace] seed-rollout convergence (per within-block pass index)")
        print(f"  {'pass':>4} | {'n_blk':>5} | {'commits':>7} | {'talk_conf':>9} | {'overlap_w_think':>15}")
        for it in sorted(self.p_n):
            n = self.p_n[it]
            ovl = self.p_ovl[it] / self.p_ovl_n[it] if self.p_ovl_n[it] else float("nan")
            print(f"  {it:>4} | {n:>5} | {self.p_commit[it]/n:>7.2f} | {self.p_conf[it]/n:>9.3f} | {ovl:>15.3f}")
        bp = self.block_passes or [0]
        print("-" * 78)
        print(f"[trace] blocks={self.n_blocks}  passes/block: mean={st.mean(bp):.1f} max={max(bp)}  "
              f"capped(hit max_iters)={self.n_capped}/{self.n_blocks} ({self.n_capped/max(1,self.n_blocks):.1%})")
        print(f"[trace] adjacent-repeat frac/block: mean={st.mean(self.block_rep or [0]):.3f}  "
              f"no-EOS seqs={self.n_no_eos}/{self.n_seq} ({self.n_no_eos/max(1,self.n_seq):.1%})")
        # heuristic verdict
        its = sorted(self.p_n)
        early = [i for i in its if i < max(1, len(its)//3)]; late = [i for i in its if i >= 2*len(its)//3]
        def _m(d, keys, dn=None):
            num = sum(d[i] for i in keys); den = sum((dn or self.p_n)[i] for i in keys)
            return num/den if den else float("nan")
        ovl_e = _m(self.p_ovl, early, self.p_ovl_n); ovl_l = _m(self.p_ovl, late, self.p_ovl_n)
        commits_l = _m(self.p_commit, late)
        rep = st.mean(self.block_rep or [0])
        print("-" * 78)
        print(f"[trace] signals: overlap early={ovl_e:.3f} -> late={ovl_l:.3f} (DRIFT if falling); "
              f"late commits/pass={commits_l:.2f} (STARVATION if ~1); rep={rep:.3f} (DEGENERACY if high)")


# ---- proven decode helpers (copied from dinfer.decoding.generate_t3d) ----------


# ---- proven decode helpers (copied from dinfer.decoding.generate_t3d) ----------
def build_block_causal_mask(L, block_length, dtype, device):
    idx = torch.arange(L, device=device)
    q_block = (idx // block_length).unsqueeze(1)
    kv_block = (idx // block_length).unsqueeze(0)
    allowed = (kv_block <= q_block)
    mask = torch.zeros(1, 1, L, L, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


def dmax_commit_uniform(logits, mask_index, active_index, threshold, fallback=True):
    """Left-to-right high-confidence prefix commit + 0.9 early-stop (DMax rule).
    fallback=True (DMax default): if NO masked position clears `threshold`, still commit the first
    masked position (guarantees >=1 commit/pass -> a block always finishes). fallback=False: commit
    ONLY the genuine >=threshold prefix, committing NOTHING when the leftmost masked position is
    below threshold (used for the think-commit hand-off: think commits only what it's confident about
    and leaves the uncertain tail to talk)."""
    x0 = logits.argmax(dim=-1)
    probs = F.softmax(logits.float(), dim=-1)
    max_probs = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    confidence = torch.where(mask_index, max_probs, torch.full_like(max_probs, -float("inf")))
    is_low_conf = mask_index & (confidence < threshold)
    has_failed = torch.cumsum(is_low_conf.long(), dim=1) > 0
    candidates = mask_index & (~has_failed)
    if fallback:
        batch_has_sel = candidates.any(dim=-1, keepdim=True)
        mask_cumsum = torch.cumsum(mask_index.long(), dim=1)
        first_mask = (mask_cumsum == 1) & mask_index
        high_conf = torch.where(batch_has_sel, candidates, first_mask)
    else:
        high_conf = candidates
    breakflag = bool(active_index.any() and (max_probs[active_index] >= 0.9).all().item())
    return x0, high_conf, max_probs, breakflag


# ---- the anchor-free top-K decode ---------------------------------------------
@torch.no_grad()
def decode_topk_talk(think, talk, emb, mask_id, prompt_ids, *, gen_length, block_length,
                     threshold, top_k, max_iters, device, dtype, keep_mask_residual=True,
                     seed_passes=1, early_stop=False, eos_id=None, trace=None,
                     think_commit_threshold=0.0, soft_commit=False):
    """The H1 inference dynamic. Per block, two phases:

      THINK-COMMIT (request 1; only if think_commit_threshold>0): think iterates and COMMITS its own
        confident left-to-right prefix at `think_commit_threshold` (e.g. 0.6), with NO fallback, until
        a pass commits nothing new (think has exhausted what it's that-confident about). The uncertain
        tail is left masked for talk. think's last logits seed talk's first pass. (think_commit_threshold
        =0 -> legacy seed: think only SEEDS `seed_passes` talk passes with candidates, never commits.)
      TALK: talk iterates the still-masked tail -- masked positions fed the top-K soft-embed of the
        current source logits (think's, then the talk's own), committing its confident prefix (DMax
        rule, WITH fallback so a block always finishes), 0.9/no-change Breakflag, until the block is full.

    soft_commit (request 2): feed COMMITTED positions to the model as the DMax soft top-K(+mask-residual)
      blend of their logits -- matching decode_uniform (parallel_strategy.py:597,662: committed = soft_cond
      gets soft_embeds) -- instead of the hard token embedding. Keeps the committed region 'alive'/revisable
      so the candidate set can still shift across passes.

    With early_stop+eos_id, stop generating further blocks once a block commits EOS (DMax
    generate_uniform's sequence-level stop; batch-filtering is a no-op at batch=1).
    Returns (response_ids, think_fwd, talk_fwd)."""
    P = prompt_ids.shape[1]
    L = ((P + gen_length + block_length - 1) // block_length) * block_length
    x = torch.full((1, L), mask_id, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids
    attn = build_block_causal_mask(L, block_length, dtype, device)
    pos = torch.arange(L, device=device).unsqueeze(0)
    first_b, num_b = P // block_length, L // block_length
    think_fwd = talk_fwd = 0
    sp = max(1, seed_passes)

    for b in range(first_b, num_b):
        bs, be = b * block_length, (b + 1) * block_length
        m, p = attn[:, :, :be, :be], pos[:, :be]

        def block_input(src_logits, feed_masked_soft):
            """Build the talk/think input. committed positions: DMax soft blend if soft_commit else hard
            token. masked positions: top-K candidate soft if feed_masked_soft (talk) else bare [MASK]
            (think generates). src_logits = the block [1,B,V] whose top-K we blend; None -> all hard."""
            inp = emb(x[:, :be]).clone()
            if src_logits is None:
                return inp
            blk = inp[:, bs:be]
            mi = (x[:, bs:be] == mask_id)
            soft = build_topk_soft_embeds(src_logits, emb, mask_id, top_k=top_k,
                                          keep_mask_residual=keep_mask_residual)
            if soft_commit:
                blk[~mi] = soft[~mi].to(inp.dtype)             # committed -> DMax soft mix
            if feed_masked_soft:
                blk[mi] = soft[mi].to(inp.dtype)               # masked -> top-K candidates
            return inp

        prev_logits = None                                     # source for the soft blend (think seed / talk prev)
        think_topk = None                                      # think's seed top-K ids [1,B,K] (drift ref)
        passes = 0

        # ---- THINK-COMMIT PHASE (request 1) ----
        if think_commit_threshold > 0:
            for _ in range(max_iters):
                mask_index = (x[:, bs:be] == mask_id)
                if not bool(mask_index.any()):
                    break
                th_logits = think(inputs_embeds=block_input(prev_logits, feed_masked_soft=False),
                                  attention_mask=m, position_ids=p, use_cache=False,
                                  return_dict=True).logits[:, bs:be]
                think_fwd += 1
                prev_logits = th_logits
                if trace is not None:
                    think_topk = th_logits.topk(top_k, dim=-1).indices
                x0, high_conf, _, _ = dmax_commit_uniform(th_logits, mask_index, mask_index,
                                                          think_commit_threshold, fallback=False)
                if not bool(high_conf.any()):                  # think exhausted its >=thr confidence -> talk
                    break
                x[:, bs:be] = torch.where(high_conf, x0, x[:, bs:be])

        # ---- TALK PHASE ----
        for it in range(max_iters):
            block_x = x[:, bs:be]
            mask_index = (block_x == mask_id)
            if not bool(mask_index.any()):
                break
            if think_commit_threshold == 0 and it < sp:        # legacy seed: think provides candidates
                th_logits = think(inputs_embeds=emb(x[:, :be]), attention_mask=m, position_ids=p,
                                  use_cache=False, return_dict=True).logits[:, bs:be]
                think_fwd += 1
                prev_logits = th_logits
                if trace is not None:
                    think_topk = th_logits.topk(top_k, dim=-1).indices
            talk_logits = talk(inputs_embeds=block_input(prev_logits, feed_masked_soft=True),
                               attention_mask=m, position_ids=p, use_cache=False,
                               return_dict=True).logits[:, bs:be]
            talk_fwd += 1
            prev_logits = talk_logits                          # feed the talk's own top-K forward
            x0, high_conf, max_probs, breakflag = dmax_commit_uniform(talk_logits, mask_index, mask_index, threshold)
            committed = high_conf & mask_index                 # newly committed this pass
            x[:, bs:be] = torch.where(high_conf, x0, block_x)
            passes += 1
            if trace is not None:
                n_commit = int(committed.sum())
                mean_conf = float(max_probs[mask_index].mean()) if bool(mask_index.any()) else float("nan")
                if n_commit and think_topk is not None:        # overlap of committed tokens with think's top-K
                    in_topk = (think_topk[0] == x0[0].unsqueeze(-1)).any(-1)   # [B] bool
                    ovl = float(in_topk[committed[0]].float().mean())
                else:
                    ovl = float("nan")
                trace.add_pass(it, n_commit, mean_conf, ovl)
            if breakflag:
                break
        if trace is not None:
            blk = x[:, bs:be]
            rep = float((blk[0, 1:] == blk[0, :-1]).float().mean()) if block_length > 1 else 0.0
            trace.add_block(passes, capped=bool((blk == mask_id).any()) or passes >= max_iters, rep=rep)
        if early_stop and eos_id is not None and bool((x[:, bs:be] == eos_id).any()):
            if be < L:                                          # EOS in this block -> rest of seq is done
                x[:, be:] = eos_id
            break
    if trace is not None:
        trace.add_seq(no_eos=(eos_id is not None and not bool((x[:, P:P + gen_length] == eos_id).any())))
    return x[:, P:P + gen_length], think_fwd, talk_fwd


# ---- think-as-DECODER family (ported from t3d_probe_converged_teacher.mixed_converge) ----
def _schedule(mode, think_seed_count, think_per_cycle=1, talk_per_cycle=1):
    """Return schedule(i)->'think'|'talk' for the think-as-decoder modes."""
    if mode == "cross":            # think->talk->think->talk... (Method D)
        return lambda i: "think" if i % 2 == 0 else "talk"
    if mode == "think_then_talk":  # think x N (commit), then talk... (Method C, N default 2)
        n = max(1, think_seed_count)
        return lambda i: "think" if i < n else "talk"
    if mode == "cycle":            # REPEATING (think x A, talk x B): A=2,B=1 -> think,think,talk,think,think,talk...
        a = max(1, think_per_cycle)
        period = a + max(1, talk_per_cycle)
        return lambda i: "think" if (i % period) < a else "talk"
    if mode == "think_only":       # pure DMax baseline (think decoded alone to convergence)
        return lambda i: "think"
    raise ValueError(f"unknown mixed decode mode: {mode}")


@torch.no_grad()
def decode_mixed(think, talk, emb, mask_id, prompt_ids, *, schedule, gen_length, block_length,
                 threshold, top_k, max_iters, device, dtype, keep_mask_residual=True,
                 early_stop=False, eos_id=None):
    """Full-response decode where a per-block schedule(i)->'think'|'talk' picks the model for step i and
    THAT model commits its own confident left-to-right prefix (DMax rule). think step = bare [MASK] at
    undecided positions; talk step = top-K soft-embed of the PREVIOUS pass's logits (think's if the prev
    step was think, else the talk's own). Ports mixed_converge to the whole sequence, but DROPS the
    probe's extra convergence-check forward (the committed ids are the output -- no need to re-read them).

    Block-end gate:
      * early_stop=False (default): loop until the block is fully committed (no masked positions left),
        capped at max_iters (=block_length) -- the dmax prefix rule commits >=1 position/step so a
        32-token block converges in <=32 steps. This is the probe's 'until converged'.
      * early_stop=True: ALSO end the block on DMax decode_uniform's Breakflag -- all active positions
        >=0.9 OR nothing changed this step (parallel_strategy.py:578-590). Matches the seed path / DMax,
        so compute is comparable across modes.
    With early_stop+eos_id, also stop generating further blocks once a block commits EOS (DMax
    generate_uniform's sequence-level stop; batch-filtering is a no-op at batch=1).
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
        cand = None                                            # the previous pass's block logits (top-K src)
        for it in range(max_iters):
            block_x = x[:, bs:be]
            mask_index = (block_x == mask_id)
            if not bool(mask_index.any()):
                break
            if schedule(it) == "think":                        # THINK decodes: bare [MASK] input, think commits
                logits = think(inputs_embeds=emb(x[:, :be]), attention_mask=m,
                               position_ids=p, use_cache=False, return_dict=True).logits[:, bs:be]
                think_fwd += 1
            else:                                              # TALK decodes: fed prev pass's top-K, talk commits
                src = cand
                if src is None:                                # block STARTS on a talk step -> 1 bootstrap think
                    src = think(inputs_embeds=emb(x[:, :be]), attention_mask=m, position_ids=p,
                                use_cache=False, return_dict=True).logits[:, bs:be]
                    think_fwd += 1
                soft = build_topk_soft_embeds(src, emb, mask_id, top_k=top_k,
                                              keep_mask_residual=keep_mask_residual)
                inp = emb(x[:, :be]).clone()
                inp[:, bs:be][mask_index] = soft[mask_index].to(inp.dtype)
                logits = talk(inputs_embeds=inp, attention_mask=m, position_ids=p,
                              use_cache=False, return_dict=True).logits[:, bs:be]
                talk_fwd += 1
            cand = logits                                      # feed this pass's top-K forward
            x0, high_conf, _, breakflag = dmax_commit_uniform(logits, mask_index, mask_index, threshold)
            new_block = torch.where(high_conf, x0, block_x)
            changed = bool((new_block != block_x).any())
            x[:, bs:be] = new_block
            if early_stop and (breakflag or not changed):      # DMax Breakflag: all>=0.9 OR no-change
                break
        if early_stop and eos_id is not None and bool((x[:, bs:be] == eos_id).any()):
            if be < L:                                          # EOS in this block -> rest of seq is done
                x[:, be:] = eos_id
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
    ap.add_argument("--seed_passes", type=int, default=1,
                    help="[seed mode] think SEEDS the first N talk passes per block (think's top-K); "
                         "thereafter the talk iterates on its OWN previous top-K (think gone). 1 = think "
                         "once/block (the H1 target), 2 = twice. think cost = N forwards/block.")
    ap.add_argument("--decode_mode", default="seed",
                    choices=["seed", "cross", "think_then_talk", "cycle", "think_only"],
                    help="seed = think-as-candidate / talk-commits (H1 dynamic, uses --seed_passes). "
                         "cross/think_then_talk/cycle/think_only = think-as-decoder family (whoever runs "
                         "commits; ported from t3d_probe_converged_teacher.mixed_converge, capped at "
                         "--max_iters/block). cross = think->talk->think->talk; think_then_talk = think x N "
                         "then talk (N=--think_seed_count); cycle = REPEATING (think x A, talk x B) per cycle "
                         "(A=--think_per_cycle, B=--talk_per_cycle; e.g. 2,1 = think,think,talk repeating); "
                         "think_only = pure DMax baseline.")
    ap.add_argument("--think_seed_count", type=int, default=2,
                    help="[think_then_talk mode] number of leading think COMMIT passes before talk takes over.")
    ap.add_argument("--think_per_cycle", type=int, default=2,
                    help="[cycle mode] think COMMIT passes per repeating cycle (the 'A' in think x A, talk x B).")
    ap.add_argument("--talk_per_cycle", type=int, default=1,
                    help="[cycle mode] talk COMMIT passes per repeating cycle (the 'B' in think x A, talk x B).")
    ap.add_argument("--early_stop", action="store_true",
                    help="Apply DMax's full early termination: per-block end on all-active>=0.9 OR no-change "
                         "(decode_uniform Breakflag), AND stop generating further blocks once EOS is committed "
                         "(generate_uniform seq stop). For 'seed' the 0.9 gate is already always on; this flag "
                         "adds the EOS stop. For the mixed modes it adds BOTH (else they run each block to full "
                         "converge). Turn ON for compute comparable to seed/DMax.")
    ap.add_argument("--eos_id", type=int, default=156892, help="EOS token id (DMax LLaDA-2.0 = 156892).")
    ap.add_argument("--trace", action="store_true",
                    help="[seed mode only] collect a rollout-convergence trace (per-pass commits / talk "
                         "confidence / overlap-with-think-seed, per-block passes+repetition, no-EOS) to pin "
                         "the talk self-rollout collapse mode (starvation / drift / degeneracy).")
    ap.add_argument("--think_commit_threshold", type=float, default=0.0,
                    help="[seed mode] >0 turns on the think-commit hand-off: think commits its own >=thr "
                         "confident prefix (no fallback) until it stalls, then talk does the uncertain tail. "
                         "0 = legacy seed (think only seeds candidates, never commits). Try 0.6.")
    ap.add_argument("--soft_commit", action="store_true",
                    help="[seed mode] feed COMMITTED positions as the DMax soft top-K(+mask-residual) blend "
                         "(decode_uniform's committed=soft_cond behavior) instead of the hard token embedding "
                         "-- keeps the committed region revisable.")
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
    sched = None if args.decode_mode == "seed" else _schedule(
        args.decode_mode, args.think_seed_count, args.think_per_cycle, args.talk_per_cycle)
    trace = RolloutTrace() if (args.trace and sched is None) else None
    if args.trace and sched is not None:
        print("[eval] --trace only applies to --decode_mode seed; ignoring.")
    print(f"[eval] {len(rows)} problems  mode={args.decode_mode} gen={args.gen_length} "
          f"block={args.block_length} top_k={args.top_k}")

    n_ok = tot_think = tot_talk = 0
    t0 = time.time()
    for i, row in enumerate(rows):
        messages = [{"role": "user", "content": GSM8K_USER_TEMPLATE.format(question=row["question"])}]
        prompt_ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=True, return_tensors="pt").to(args.device)
        if sched is None:                                      # seed mode: think-as-candidate / talk-commits
            resp, th, tk = decode_topk_talk(think, talk, emb, mask_id, prompt_ids,
                                            gen_length=args.gen_length, block_length=args.block_length,
                                            threshold=args.threshold, top_k=args.top_k,
                                            max_iters=args.max_iters, device=args.device, dtype=dtype,
                                            keep_mask_residual=not args.no_mask_residual,
                                            seed_passes=args.seed_passes,
                                            early_stop=args.early_stop, eos_id=args.eos_id, trace=trace,
                                            think_commit_threshold=args.think_commit_threshold,
                                            soft_commit=args.soft_commit)
        else:                                                  # think-as-decoder family (cross / think_then_talk / think_only)
            resp, th, tk = decode_mixed(think, talk, emb, mask_id, prompt_ids, schedule=sched,
                                        gen_length=args.gen_length, block_length=args.block_length,
                                        threshold=args.threshold, top_k=args.top_k,
                                        max_iters=args.max_iters, device=args.device, dtype=dtype,
                                        keep_mask_residual=not args.no_mask_residual,
                                        early_stop=args.early_stop, eos_id=args.eos_id)
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
