# Copyright 2026 University of Sydney. Apache-2.0.
"""G1 (DBet x SGLang) — extract the drafter's heavy features (h_sel, h_last) from the SGLang diffusion
heavy via forward HOOKS, without modifying the vendored sglang model.

Why hooks + why (hidden+residual):
  The sglang LLaDA2Moe model carries a SPLIT residual stream — each decoder layer takes/returns
  (hidden_states, residual) and folds them lazily in `layer_communicator.prepare_attn`
  (modeling_llada2_moe_sglang.py:990). The eager `extract_heavy_signals` hidden the drafter was TRAINED on
  is the fully-added hidden `out.hidden_states[k]`. At a layer boundary that equals `hidden + residual`.
  A forward hook on sglang layer i receives the layer OUTPUT `(hidden, residual, present_kv)`, so
    (out[0] + out[1])  ==  eager hidden_states[i+1].
  Therefore eager hs[k] = (h+r) AFTER sglang layer (k-1). For `config.sel_layers_list = [1,10,19]` we hook
  layers [0,9,18]; h_last = hs[num_layers] -> hook the last decoder layer.

Scope: hooks fire in eager execution only. With CUDA graphs ON, a graph REPLAY does not run Python hooks —
so use `--disable-cuda-graph` for now (plan Phase 2); Phase 3 replaces this with graph-captured buffers.

Validation (REQUIRED before trusting the drafter downstream): run the SAME block input through both this tap
(sglang) and eager `DbetForDraftDecoding.extract_heavy_signals`, and assert h_sel/h_last match (fp32: ~1e-4;
bf16: within rounding, like the cache saga). Structural mismatch (wrong capture point / residual convention)
would feed the drafter out-of-distribution features.
"""
import torch


class HeavyFeatureTap:
    """Registers forward hooks on a sglang decoder-layer ModuleList to capture the DBet drafter features.

    Usage:
        tap = HeavyFeatureTap(decoder_layers, sel_layers=cfg.sel_layers_list, num_layers=len(decoder_layers))
        ... model_runner.forward(forward_batch) ...          # hooks fill the buffer during the forward
        h_sel, h_last = tap.pop()                            # [B, seq, m*D], [B, seq, D] for that forward
        ...
        tap.remove()                                         # detach hooks when done
    `decoder_layers` = the heavy's decoder-layer ModuleList (e.g. runner.model.model.layers). Each layer's
    forward must return a tuple whose first two elements are (hidden_states, residual)."""

    def __init__(self, decoder_layers, sel_layers, num_layers, final_norm=None):
        self.sel_layers = list(sel_layers)                  # eager hs indices, e.g. [1, 10, 19] (intermediate, pre-norm)
        self.h_last_hs = num_layers                         # eager hs index of h_last
        self._buf = {}                                      # {eager_hs_index: [B, seq, D]}
        self._handles = []
        # h_sel: hook sglang layer (k-1) -> (hidden+residual) == eager hidden_states[k] (pre-norm intermediate)
        want = {k - 1: k for k in self.sel_layers}
        for i, layer in enumerate(decoder_layers):
            if i in want:
                self._handles.append(layer.register_forward_hook(self._make_hidden_hook(want[i])))
        # h_last: eager `hidden_states[-1]` is POST-final-norm (the eager model applies self.norm THEN appends).
        # So hook the final-norm OUTPUT, not the last decoder layer (which is pre-norm -> would blow up rel-Δ).
        if final_norm is not None:
            self._handles.append(final_norm.register_forward_hook(self._make_norm_hook(self.h_last_hs)))
        else:  # fallback: pre-norm last-layer hidden (ONLY correct if the eager h_last is pre-norm)
            self._handles.append(decoder_layers[num_layers - 1].register_forward_hook(
                self._make_hidden_hook(self.h_last_hs)))
        missing = [k for k in self.sel_layers if (k - 1) not in range(num_layers)]
        if missing:
            raise ValueError(f"sel indices {missing} out of range for num_layers={num_layers}")

    def _make_hidden_hook(self, hs_index):
        def hook(_module, _inp, out):
            # decoder-layer out = (hidden_states, residual, present_kv); eager hs = hidden + residual
            h = out[0]
            r = out[1] if isinstance(out, tuple) and len(out) > 1 else None
            self._buf[hs_index] = (h + r if r is not None else h).detach()
        return hook

    def _make_norm_hook(self, hs_index):
        def hook(_module, _inp, out):
            # final-norm out = post-norm hidden ([B,seq,D] tensor, or a (hidden, ...) tuple)
            self._buf[hs_index] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    def pop(self):
        """(h_sel [B,seq,m*D], h_last [B,seq,D]) captured by the most recent forward; clears the buffer."""
        try:
            h_sel = torch.cat([self._buf[k] for k in self.sel_layers], dim=-1)
            h_last = self._buf[self.h_last_hs]
        except KeyError as e:
            raise RuntimeError(f"HeavyFeatureTap: feature {e} not captured — did a forward run with hooks "
                               f"active (CUDA graphs must be OFF)?")
        self._buf = {}
        return h_sel, h_last

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []
