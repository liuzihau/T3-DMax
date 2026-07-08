# Copyright 2026 University of Sydney. Apache-2.0.
"""Phase 3 (SGLang x DBet, the OPTIMIZED path) — inject the eager drafter into DMax's WORKING cached + CUDA-graph
block-diffusion decode (BlockDiffusionLLM), instead of rebuilding the cache/graph loop.

Seam: BlockDiffusionIteration.forward_uniform does one HEAVY forward (block-only, prefix in the KV cache) + a
decode_uniform commit, returning (output, Breakflag, embeddings) where `embeddings` is the soft-embed feed for the
next forward. DbetBlockDiffusionIteration overrides it to, right after the commit:
  1) pop the block's h_sel/h_last from the model's graph-safe buffer tap (modeling_llada2_moe_sglang_dbet),
  2) run the compiled DRAFTER on (canvas = block, prefix = the drafter's cross-block prefix cache),
  3) EXTEND (commit confident still-masked slots) + FIX (override committed) -> write into x.data + soft-embed the
     draft commits into `embeddings` for the next heavy forward.
On Breakflag (block settled) it EXTENDS the drafter prefix cache with the just-settled block's h_sel.

The heavy runs under CUDA graphs + prefix cache (the 927 tok/s path); the buffer copy_ tap survives graph replay.
STATUS: first integration draft -- needs box iteration (TokenArray / embeddings shapes / cross-block path / the
tap length under the cached forward are the coupling points flagged inline).
"""
import torch

from dinfer.decoding.generate_uniform import BlockDiffusionIteration, BlockDiffusionLLM
from dinfer.decoding.generate_dbet import _soft_embed, MASK_ID


class DbetBlockDiffusionIteration(BlockDiffusionIteration):
    """BlockDiffusionIteration + a DRAFTER step after each heavy commit. `draft` = standalone DbetDraftStack
    (load_drafter_standalone), `heavy_model` = the LLaDA2Model with enable_dbet_tap() on (the graph-safe buffer tap).
    `draft_cache` = the drafter's cross-block prefix KV cache (grown per settled block)."""

    def __init__(self, draft, heavy_model, draft_threshold=0.9, draft_tau=1.0, draft_top_k=2, draft_fix=True,
                 draft_enabled=True):
        super().__init__()
        self.draft = draft
        self.heavy_model = heavy_model                 # runner.model.model (LLaDA2Model with the buffer tap)
        self.embed = draft.frozen_embed if draft is not None else None   # draft=None -> heavy-only baseline
        self.draft_enabled = draft_enabled             # False -> pure heavy-only decode (baseline; the EXTEND's
        self.draft_threshold = draft_threshold         #   keep[0]=True progress rule means threshold can't disable it)
        self.draft_tau = draft_tau
        self.draft_top_k = draft_top_k
        self.draft_fix = draft_fix
        # FIXED-size prefix KV (static shape -> torch.compile fast path; a growing DynamicCache made the draft
        # forward a new shape each block -> ~eager 3.9ms instead of the compiled 1.5ms floor).
        self.max_prefix = 640                          # >= max settled gen tokens (gen_length up to ~600)
        if draft is not None:
            self.draft_cache = draft.new_fixed_prefix_cache(self.max_prefix, draft.frozen_embed.weight.device)
        else:
            self.draft_cache = None
        self.draft_commits = 0

    def reset(self):
        if self.draft_cache is not None:
            self.draft_cache.reset()
        self.draft_commits = 0

    @torch.no_grad()
    def forward_uniform(self, model, decoder, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask,
                        past_key_values, replace_position, backend, active_index, embeddings, embedding_layer,
                        is_cross_block=False, block_length=32):
        # 1) HEAVY forward (block, prefix-cached) + decode_uniform commit -- unchanged DMax
        output, Breakflag, embeddings = super().forward_uniform(
            model, decoder, x, kv_cache, block, block_loc, block_id, pos_ids, attn_mask, past_key_values,
            replace_position, backend, active_index, embeddings, embedding_layer,
            is_cross_block=is_cross_block, block_length=block_length)

        if not self.draft_enabled:                                       # baseline: pure DMax heavy decode
            return output, Breakflag, embeddings

        be = block_loc.end
        blk = block_length                                # the REAL current block (cross-block passes a 2-block span)
        cur = be - blk                                    # current-block start (== block_loc.start when not cross)
        B = x.batch_size
        # 2) pop the tapped features. The forward spans block_loc.end-block_loc.start (= blk normally, 2*blk on the
        #    cross-block first step); the CURRENT block is always the LAST blk rows.
        qlen = be - block_loc.start
        h_sel_all, h_last_all = self.heavy_model.pop_dbet_features(B, qlen)
        h_sel = h_sel_all[:, -blk:]; h_last = h_last_all[:, -blk:]        # current block
        block_logits = output.logits[:, -blk:]                           # [B, blk, V]

        active = active_index[:, -blk:] if active_index.shape[1] != blk else active_index

        if bool(Breakflag):
            # block settled -> extend the drafter prefix cache with this block's (settled) h_sel for later blocks
            pos = torch.arange(cur, be, device=x.device).unsqueeze(0)
            self.draft.extend_prefix_cache(h_sel, pos, self.draft_cache)
            self.settled = be
            return output, Breakflag, embeddings

        # 3) DRAFT step (canvas = current block, prefix = cross-block draft cache)
        block_x = x.data[0, cur:be]
        mask_pos = (block_x == MASK_ID)
        committed_before = active[0] & (~mask_pos)
        draft_ids = x.data[:, cur:be].clone()
        settled = self.draft_cache.settled
        M = self.draft_cache.max_len
        # STATIC-shape inputs so torch.compile stays on its fast path (values vary per block, SHAPES don't):
        pos = torch.arange(settled, settled + blk, device=x.device).unsqueeze(0)       # canvas positions (== G2)
        pmask = torch.zeros(1, 1, 1, M + blk, dtype=torch.bool, device=x.device)        # prefix mask (hides pad)
        pmask[..., :settled] = True                                                     # valid settled prefix
        pmask[..., M:] = True                                                           # canvas (bidirectional)
        d = self.draft(input_ids=draft_ids, heavy_logits=block_logits, h_sel_denoise=h_sel, h_last_denoise=h_last,
                       h_sel_prefix=None, past_key_values=self.draft_cache,
                       attention_mask=pmask, position_ids=pos, denoise_mask=None, tau=self.draft_tau)
        dlogits, dconf = d["logits"], d["conf"]
        if dconf is None:
            return output, Breakflag, embeddings
        darg = dlogits[0].argmax(-1); dc = dconf[0]

        # EXTEND: left-to-right prefix commit of masked slots while conf >= threshold (>=1 for progress)
        mloc = mask_pos.nonzero(as_tuple=True)[0]
        if mloc.numel() > 0:
            ok = dc[mloc] >= self.draft_threshold
            keep = ~(torch.cumsum((~ok).long(), 0) > 0); keep[0] = True
            sel = mloc[keep]
            x.data[0, cur + sel] = darg[sel]
            embeddings[0, sel] = _soft_embed(dlogits[0][sel], self.embed, MASK_ID, self.draft_tau, self.draft_top_k)
            self.draft_commits += int(sel.numel())
        # FIX: override a committed slot iff conf-head >= threshold AND the draft disagrees
        if self.draft_fix and bool(committed_before.any()):
            fix = committed_before & (dc >= self.draft_threshold) & (darg != block_x)
            floc = fix.nonzero(as_tuple=True)[0]
            if floc.numel() > 0:
                x.data[0, cur + floc] = darg[floc]
                embeddings[0, floc] = _soft_embed(dlogits[0][floc], self.embed, MASK_ID, self.draft_tau, self.draft_top_k)
        return output, Breakflag, embeddings


class DbetBlockDiffusionLLM(BlockDiffusionLLM):
    """BlockDiffusionLLM with the DBet drafter injected. Pass `draft` (standalone DbetDraftStack) + `sel_layers`;
    call enable the model's buffer tap once. Everything else (cache, cuda graphs, prefix) is DMax's, untouched."""

    def __init__(self, model, decoder, iterator_factory, cache_factory, draft, sel_layers, *,
                 draft_threshold=0.9, draft_tau=1.0, draft_top_k=2, draft_fix=True, draft_enabled=True,
                 early_stop=True, maximum_unroll=1, expected_tpf=15, backend='sglang', **kw):
        super().__init__(model, decoder, iterator_factory, cache_factory, early_stop=early_stop,
                         maximum_unroll=maximum_unroll, expected_tpf=expected_tpf, backend=backend, **kw)
        # turn on the graph-safe buffer tap on the inner LLaDA2Model (runner.model = LLaDA2SGLangLM; .model = LLaDA2Model)
        heavy_model = model.model.model
        # buffer covers a block forward (<= 2*block_length on cross-block); the long prompt prefill exceeds it and is
        # skipped by the model's copy_ guard (we only need per-block features, never prefill). MUST be enabled BEFORE
        # the runner captures its CUDA graph -> the driver enables it pre-ModelRunner; skip re-alloc if already on
        # (re-allocating would orphan the buffers the graph captured).
        if draft is not None and heavy_model._dbet is None:
            heavy_model.enable_dbet_tap(sel_layers, max_bs=1, max_len=256)
        self.diff_iteration = DbetBlockDiffusionIteration(
            draft, heavy_model, draft_threshold=draft_threshold, draft_tau=draft_tau,
            draft_top_k=draft_top_k, draft_fix=draft_fix, draft_enabled=draft_enabled)
        # rebuild the runner around the DBet iteration (BlockDiffusionRunner holds a ref to diff_iteration)
        self.block_runner.diff_iteration = self.diff_iteration
