# DBet × SGLang — batched inference plan

**Status:** G1 ✅ + G2 ✅ DONE (2026-07-07) — sglang-DBet runs end-to-end and is ISO-ACCURATE (GSM8K n=100 = 89.0%
vs eager-DBet 89.99%). Files: `decoding/generate_dbet_sglang.py`, `evaluations/eval_dbet_sglang_gsm8k.py`,
`decoding/dbet_sglang_features.py` (tap), `load_drafter_standalone`. **NEXT = BATCHING** (the actual speedup;
batch=1 has no win yet) → lm-eval → CUDA-graph tap. **Owner:** nick + Claude. **Goal:** get DBet's drafter-offload speedup
to actually materialize by moving the heavy into **batched, compute-bound** serving (SGLang), since batch=1
decode is memory-bandwidth-bound and neither the KV-cache nor the drafter offload converts to wall-clock there.

---

## 0. Why (the premise)

At **batch=1**, a forward of the 16B MoE heavy is dominated by **loading weights from HBM**, independent of token
count. Consequences we measured:
- The prefix-KV cache is *correct* but ~marginal / net-negative at batch=1 (weight load unchanged; KV-copy overhead
  added). See `experiment.md §cache`.
- The drafter is cheap (4–6 ms vs 31–33 ms heavy) and offloads ~40% of commits, but the heavy forwards it *saves*
  were nearly free (overlapped with weight streaming), so wall-clock barely moves.

**Batching amortizes the weight load across N sequences → compute-bound regime**, where (a) the drafter offload and
(b) fewer/cheaper heavy forwards finally convert to throughput. SGLang gives batching + paged KV + fused kernels +
CUDA graphs. **This premise MUST be validated before the big build (Phase 0).**

---

## 1. What already exists (the hard 80%)

dInfer already runs the **diffusion heavy** in SGLang:
- `python/dinfer/model/modeling_llada2_moe_sglang.py` (1735 ln): LLaDA2Moe heavy as a native SGLang model.
  - `H2Embed` (~ln 235): logits → softmax → **continuous/soft embedding** re-feed (the DMax decode_uniform input).
  - top-level `forward` (~ln 1204) → `MoeCausalLMOutputWithPast(logits, hidden_states=<final only>)`; supports
    `inputs_embeds` (~ln 1208). Layer loop at ~ln 1116 (this is where intermediate hiddens must be tapped).
- `python/dinfer/into_sglang/algorithm.py`: `dInferDiffusionAlgorithm.run` — drives block-diffusion decode through
  `model_runner.forward(forward_batch)`, reshapes `full_logits` → [B, block, V], calls `decoder.decode(...)` to commit.
- `evaluations/eval_dinfer_sglang.py`: builds `ModelRunner` + `ThresholdParallelDecoder` (commit rule) +
  `BlockDiffusionLLM(..., backend='sglang')`, runs batched lm-eval GSM8K. **This is the template to copy.**

So: batched block-diffusion + soft-embed re-feed + threshold commit + paged KV all work for the pure heavy.

## 2. What DBet adds (the eager reference)

Eager DBet = `python/dinfer/decoding/generate_dbet.py::decode_block_dbet` + `dFactory/models/dbet/modeling_dbet.py`.
Per block iter it does: heavy forward → `extract_heavy_signals` (**logits, h_sel, h_last**) → `dmax_commit_uniform`
→ drafter forward (`model.draft`) gated by the conf head → drafter EXTEND/FIX commits → soft-embed re-feed. The
drafter needs, per step, from the heavy:
- `logits` [B, blk, V]  — **already produced** (full_logits).
- `h_last` [B, blk, D]  — the final hidden — **already returned**.
- `h_sel` = concat of hidden at `config.sel_layers_list` (**[1, 10, 19]**) → [B, blk, 3·D] — **NOT exposed by SGLang**.
- prefix `h_sel` (committed context) for the drafter's prefix-KV — same, for prefix positions (or via a per-seq
  cross-block draft cache, which the eager model already implements: `extend_prefix_cache` / `cache_prefix`).

Back to the heavy: the drafter's committed tokens join the commit set and are re-fed as **soft-embeds** to the next
heavy forward (same H2Embed mechanism, drafter logits/positions added).

---

## 3. Architecture decision — Option A (heavy in SGLang, drafter eager)

**SGLang runs ONLY the heavy** (batched, where the win is). **The drafter runs eager** in a DBet block-loop variant
that mirrors `decode_block_dbet` but calls the SGLang heavy for the forward, consuming its extracted features and
feeding commits back. Rejected: **Option B** (drafter *inside* SGLang with its own paged KV / attention backend /
CUDA graph) — weeks more work, no early payoff.

Seam: a `DbetDiffusionAlgorithm` (parallel to `dInferDiffusionAlgorithm`) or a `backend='sglang'` DBet decode that,
per block iter: SGLang heavy forward (returns logits + tapped h_sel/h_last) → heavy commit → **eager drafter** on the
features (batched [B, blk, *], per-seq conf-gated commits) → update block state + soft-embed → next iter.

---

## 4. The three gaps

| # | Gap | Approach | File / anchor | Difficulty |
|---|-----|----------|---------------|------------|
| G1 | Expose `h_sel` (layers 1/10/19) + prefix `h_sel` from the SGLang heavy | Tap the layer loop, stash hidden after the sel layers into the output struct. **CUDA-graph-safe = write to preallocated buffers**; start with `--disable-cuda-graph` for correctness. | `modeling_llada2_moe_sglang.py` layer loop ~ln 1116, output ~ln 1238 | **Medium** |
| G2 | Run the drafter on extracted features, batched, per-seq commit gating | Reuse `DbetForDraftDecoding.draft` + the cross-block draft cache eager; make it batched with a **per-sequence** prefix cache + commit/EOS bookkeeping. | new `generate_dbet_sglang.py`; drafter from `modeling_dbet.py` | **Easy–Med** |
| G3 | Feed drafter commits back as soft-embeds for the next heavy forward | Add drafter-committed positions to the commit set; route drafter logits through the H2Embed soft-embed path; update the paged prefix. | `into_sglang/algorithm.py` + `H2Embed` | **Medium** |

Note on "so many features": volume is NOT the problem — `logits` (150k×blk) already moves; `h_sel` is 3·D (a few
MB/step). G1 is *plumbing to expose intermediate layers*, not bandwidth.

### G1 — STARTED (2026-07-07): `decoding/dbet_sglang_features.py::HeavyFeatureTap`
**Approach = forward hooks, NO model-file surgery.** The sglang heavy carries a SPLIT residual stream
(`hidden_states`, `residual`; folded in `layer_communicator.prepare_attn`, modeling_llada2_moe_sglang.py:990).
The eager hidden the drafter was trained on (`extract_heavy_signals` → `out.hidden_states[k]`) = the fully-added
hidden, which at a layer boundary is **`hidden + residual`**. A forward hook on sglang layer `i` gets output
`(hidden, residual, kv)`, so **`(out[0]+out[1]) == eager hidden_states[i+1]`**. ⇒ hook sglang layers `{k-1 : k in
sel_layers}` for h_sel and the LAST layer for h_last (`hs[num_layers]`, pre-final-norm). Hidden is `[B, seq, D]`
(this dInfer adaptation keeps 3-D, not SGLang's flat packing), so `h_sel = cat(sel, dim=-1)` is `[B, seq, m·D]`.
Hooks fire in eager only → **CUDA graphs OFF** (Phase 3 = graph-captured buffers). Wire: build the tap on the
heavy's decoder-layer ModuleList (`runner.model.model.layers` or wherever it resolves), `tap.pop()` after each
`model_runner.forward`.
**✅ G1 VALIDATED (2026-07-07, `evaluations/validate_sglang_hsel.py`).** bf16 (sglang kernels reject fp32),
block-0 forward, sglang `DMax-Math-16B` (per-expert) vs eager `DMax-Math-16B-moe-merge` — SAME math model, both
sides. Result: `hs1 rel=0.5%` (capture rule correct), `hs10 2.4%`, `hs19 5.2%` (bf16 compounding across two
impls), `h_last rel=6.2%` (after fixing: h_last = POST-final-norm, hook `model.norm` output not the last layer),
**logits ARGMAX_MATCH=98.4%** (the heavies compute the same fn). ⇒ the sglang heavy is feature-compatible with the
eager-trained drafter; the tap is correct.
**Gotchas hit + fixed (all in validate_sglang_hsel.py / the tap):** sglang env = `conda_env_bucket/dInfer`
(sglang 0.5.3.post1); use the ORIGINAL per-expert checkpoint (loader fuses experts; merged fails to parse) AND the
MATH model (base `DMax-16B` ≠ `DMax-Math-16B`); bf16-only (don't `.to(bf16)` — breaks the float32 rope cos_sin_cache);
logits are `out.logits` not `.full_logits`; h_last is post-final-norm.
**Residual, NOT yet proven:** whether the drafter DECODES well on ~5%-perturbed h_sel + a 98.4%-agreeing heavy —
that's the end-to-end **G2** accuracy test (expect a slightly different-but-valid trajectory, cf. the cache saga).

---

## 5. Phased plan

- **Phase 0 — premise check (1–2 d, cheap, DECISIVE).** Measure heavy-only GSM8K tok/s at batch **1 / 4 / 8 / 16**
  via the existing `eval_dinfer_sglang.py` (already batches). If tok/s scales ~linearly → compute-bound reached →
  DBet-in-SGLang will pay off → proceed. If it plateaus early → batching alone won't help; STOP and rethink.
  - Optional cheaper proxy / stepping stone: **batched *eager* DBet** (~1 wk) — make `generate_dbet` handle `B>1`
    (per-seq block state/EOS/commit). Tests the same premise without SGLang; may be "good enough."
- **Phase 1 — decide** stepping-stone (batched-eager) vs straight to Option A, based on Phase 0 + appetite.
- **Phase 2 — build Option A (2–3 wk), CUDA graphs OFF first:**
  1. G1: tap h_sel/prefix-h_sel; verify they match the eager `extract_heavy_signals` bit-for-bit (fp32) / close (bf16).
  2. G2: eager batched drafter on the features; per-seq prefix cache + conf-gated EXTEND/FIX.
  3. G3: drafter commits → soft-embed re-feed; block-state + paged-prefix update.
  4. Wire into a `DbetDiffusionAlgorithm` / `generate_dbet_sglang.py`; **validate GSM8K accuracy matches eager DBet**.
- **Phase 3 — optimize (1 wk):** re-enable CUDA graphs (G1 buffer-safe), tune batch size, measure throughput vs
  eager DBet and vs heavy-only SGLang. Report the batched drafter-offload speedup.

## 6. Risks & mitigations
- **CUDA graphs vs intermediate capture** → start graphs-off (most of the batch win is batching+kernels); make G1
  write to fixed buffers before re-enabling.
- **Batched per-seq bookkeeping** (commit/EOS/prefix cache diverge per sequence) → the fiddliest new code; lean on
  the existing batched block-diffusion algorithm; test with B=2 first.
- **Paged-KV × soft-embed** for drafter commits → mirror how the heavy's own commits already page; add drafter
  positions to the same path.
- **Premise could fail** (Phase 0) → that's the point of doing it first, cheaply.

## 7. Success criteria
1. Phase 0: heavy tok/s scales with batch (compute-bound crossover shown).
2. Phase 2: batched DBet **GSM8K accuracy == eager DBet** (within noise) at the winner config (h0.9 d0.9 k2).
3. Phase 3: batched DBet **throughput > heavy-only SGLang** at the same batch (the drafter offload converts), and
   **> eager DBet** batch=1 by a wide margin.

## 8. Estimate
Heavy-in-SGLang is done. DBet add-on: **~2–4 weeks** focused (G1 1–2 d, G2 2–3 d, G3 1–2 d, integration+correctness
3–5 d, batched edge cases 3–5 d, optimize 1–2 d). Gate the spend on Phase 0.

---
*Cross-refs: `fork_bounded_surrogate/experiment.md` (§cache, §decode), memory `dbet-kv-cache-interrupted`,
`dbet-decode-and-first-win`. Eager DBet decode = `generate_dbet.py`; cache diagnostics parked in `evaluations/attic/`.*
