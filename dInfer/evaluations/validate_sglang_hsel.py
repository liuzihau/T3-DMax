# Copyright 2026 University of Sydney. Apache-2.0.
"""G1 GATE — validate that the SGLang heavy's tapped h_sel/h_last == the eager `extract_heavy_signals`
the DBet drafter was trained on. If this passes, the sglang heavy is feature-compatible and we build G2/G3;
if it fails structurally, the drafter would get out-of-distribution features (stop and fix / retrain).

Two modes, run as SEPARATE processes (avoids a double 16B load); they share one controlled block-0 forward
via a dumped tensor file. **The sglang model is bf16-only** (its rope / fused-MoE / attention are custom
half-precision CUDA kernels — fp32 dispatch fails), so validate in bf16:
  1) SGLANG env:  python evaluations/validate_sglang_hsel.py --mode sglang --heavy_path <ORIGINAL per-expert DMax> \
         --dump /tmp/g1.pt --dtype bfloat16
  2) dFactory env: python evaluations/validate_sglang_hsel.py --mode eager --heavy_path <merged DMax> \
         --drafter_path <hf_ckpt> --dump /tmp/g1.pt --dtype bfloat16

Reading it: across two *different* implementations in bf16, absolute Δ compounds over 20 layers, so judge by
**logits ARGMAX_MATCH (~1.0 = same computation)** and the **earliest sel layer's relative Δ** (rounding-scale =
capture point correct), NOT absolute Δ. A structural bug = argmax disagreement + large earliest-layer rel-Δ.
The definitive test remains end-to-end drafter accuracy (G2). `--heavy_path` for --mode sglang must be the
ORIGINAL per-expert checkpoint (the sglang loader fuses experts itself; the merged ckpt fails to parse).

NOTE: the --mode sglang setup mirrors evaluations/eval_dinfer_sglang.py (distributed/ServerArgs/ModelRunner).
Reconcile with your working eval if the sglang API differs on your box; the eager side + the compare protocol
are the load-bearing parts.
"""
import argparse
import os
import sys
import types

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "python")))

MASK_ID = 156895
BLOCK = 32
PROMPT = "Natalia sold clips to 48 friends in April, then half as many in May. How many total?"


def exact_moe_patch(model):
    """Swap the fused experts for a row-independent eager path (fp32-capable, length-invariant). Mirrors
    evaluations/attic/diag_cache.py; matches any module with 3-D gate/up/down expert weights."""
    def _exact(self, hidden_states, expert_idx=None, routing_weights=None, selected_experts=None):
        if expert_idx is not None:
            g = torch.matmul(hidden_states, self.gate_proj[expert_idx].transpose(0, 1))
            u = torch.matmul(hidden_states, self.up_proj[expert_idx].transpose(0, 1))
            return torch.matmul(self.act_fn(g) * u, self.down_proj[expert_idx].transpose(0, 1))
        out = torch.zeros_like(hidden_states)
        for k in range(selected_experts.shape[1]):
            eids = selected_experts[:, k]; w = routing_weights[:, k]
            for e in torch.unique(eids).tolist():
                m = eids == e; hs = hidden_states[m]
                g = torch.matmul(hs, self.gate_proj[e].transpose(0, 1))
                u = torch.matmul(hs, self.up_proj[e].transpose(0, 1))
                y = torch.matmul(self.act_fn(g) * u, self.down_proj[e].transpose(0, 1))
                out[m] += w[m].unsqueeze(-1) * y
        return out
    n = 0
    for mod in model.modules():
        if all(hasattr(mod, a) for a in ("gate_proj", "up_proj", "down_proj", "act_fn")) \
           and getattr(mod, "gate_proj").dim() == 3:
            mod.forward = types.MethodType(_exact, mod); n += 1
    print(f"[exact_moe] patched {n} fused-experts modules")


def build_block0_input(tokenizer, device):
    """Deterministic block-0 input: [prompt ; MASK*BLOCK] over a block-aligned [0, be). All-hard (no soft
    embeds, no cache) -> exactly what block-0 iter-0 feeds. Returns (x [1,be], P, be)."""
    messages = [{"role": "user", "content": PROMPT + "\nLet's think step by step\n"}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_tensors="pt").to(device)
    P = prompt_ids.shape[1]
    fbs = (P // BLOCK) * BLOCK
    be = fbs + BLOCK
    x = torch.full((1, be), MASK_ID, dtype=torch.long, device=device)
    x[:, :P] = prompt_ids
    return x, P, be


def block_causal_bool(be, device):
    """Bool block-causal mask [1, be, be], True = attend (sglang format)."""
    idx = torch.arange(be, device=device)
    qb = (idx // BLOCK).unsqueeze(1); kb = (idx // BLOCK).unsqueeze(0)
    return (kb <= qb).unsqueeze(0)                                   # [1, be, be]


def block_causal_additive(be, dtype, device):
    """Additive block-causal mask [1,1,be,be], 0 = attend / -inf = mask (eager format)."""
    idx = torch.arange(be, device=device)
    qb = (idx // BLOCK).unsqueeze(1); kb = (idx // BLOCK).unsqueeze(0)
    m = torch.zeros(1, 1, be, be, dtype=dtype, device=device)
    m.masked_fill_((kb > qb).unsqueeze(0).unsqueeze(0), float("-inf"))
    return m


# ======================================================================================
def run_sglang(args):
    from transformers import AutoTokenizer, AutoConfig
    from sglang.srt.server_args import ServerArgs
    from sglang.srt import distributed
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.layers.moe import initialize_moe_config
    from dinfer.model.modeling_llada2_moe_sglang import LLaDA2SGLangLM
    from dinfer.decoding.diffusion_runner import ModelRunner
    from dinfer.decoding.dbet_sglang_features import HeavyFeatureTap

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    # --- distributed / sglang bring-up (mirror eval_dinfer_sglang.py; tp=1) ---
    os.environ.setdefault("MASTER_ADDR", "localhost"); os.environ.setdefault("MASTER_PORT", "23511")
    distributed.init_distributed_environment(1, 0, "env://", 0, "nccl")
    distributed.initialize_model_parallel(1, 1, 1, backend="nccl")
    model_config = AutoConfig.from_pretrained(args.heavy_path, trust_remote_code=True)
    server_args = ServerArgs(model_path=args.heavy_path, enable_dp_attention=True, trust_remote_code=True,
                             tp_size=1, dp_size=1, pp_size=1)
    try:
        from sglang.srt.server_args import set_global_server_args_for_scheduler
        set_global_server_args_for_scheduler(server_args)
    except ImportError:
        pass
    initialize_dp_attention(server_args=server_args, model_config=model_config)
    initialize_moe_config(server_args)
    torch.set_default_dtype(torch.bfloat16)
    model = LLaDA2SGLangLM(config=model_config, expert_map_path=".").eval()
    model.load_weights(args.heavy_path, device=device)
    model = model.to(device)
    # cuda graph OFF so the tap hooks fire on our forward
    runner = ModelRunner(model, device, enable_cuda_graph=False, server_args=server_args, max_length=512)

    DT = getattr(torch, args.dtype)
    runner.model.to(DT)
    if args.exact_moe:
        exact_moe_patch(runner.model)
    num_layers = len(runner.model.model.layers)                     # <-- reconcile if the handle differs
    sel = list(model_config.sel_layers_list) if hasattr(model_config, "sel_layers_list") else [1, 10, 19]
    tap = HeavyFeatureTap(runner.model.model.layers, sel_layers=sel, num_layers=num_layers)

    tokenizer = AutoTokenizer.from_pretrained(args.heavy_path, trust_remote_code=True)
    x, P, be = build_block0_input(tokenizer, device)
    attn = block_causal_bool(be, device)
    pos = torch.arange(be, device=device).unsqueeze(0)
    with torch.no_grad():
        out = runner.forward(input_ids=x, position_ids=pos, attention_mask=attn, use_cache=False)
    h_sel, h_last = tap.pop()
    logits = getattr(out, "full_logits", None)
    if logits is not None:
        logits = logits.view(1, be, -1)
    torch.save({"x": x.cpu(), "P": P, "be": be, "sel": sel, "num_layers": num_layers,
                "h_sel": h_sel.float().cpu(), "h_last": h_last.float().cpu(),
                "logits": None if logits is None else logits.float().cpu()}, args.dump)
    print(f"[sglang] dumped h_sel{tuple(h_sel.shape)} h_last{tuple(h_last.shape)} -> {args.dump}")
    tap.remove()


# ======================================================================================
def run_eager(args):
    from transformers import AutoTokenizer  # noqa: F401 (kept for parity / tokenizer sanity)
    from dinfer.decoding.generate_dbet import load_dbet_model
    d = torch.load(args.dump, map_location="cpu")
    device = torch.device("cuda:0")
    model = load_dbet_model(args.drafter_path, args.heavy_path, str(device))
    DT = getattr(torch, args.dtype)
    model.to(DT)
    if args.exact_moe:
        exact_moe_patch(model)

    x = d["x"].to(device); be = d["be"]; sel = d["sel"]
    attn = block_causal_additive(be, DT, device)
    with torch.no_grad():
        sig = model.extract_heavy_signals(x, attention_mask=attn)   # embeds internally (all-hard block-0 input)
    e_hsel, e_hlast, e_logits = sig["h_sel"].float(), sig["h_last"].float(), sig["logits"].float()

    s_hsel = d["h_sel"].to(device); s_hlast = d["h_last"].to(device)

    def rel(a, b):
        """(max|Δ|, mean|Δ|, mean|Δ| / mean|a|) — the relative ratio is the bf16-robust structural signal."""
        dd = (a - b).abs()
        scale = a.abs().mean().item() + 1e-9
        return dd.max().item(), dd.mean().item(), dd.mean().item() / scale

    print(f"[eager vs sglang]  dtype={args.dtype}  sel_layers={sel}  (sglang is bf16-only: fp32 unsupported)")
    # per-sel-layer h_sel: split [B,seq,m*D] into the m sel layers; hs[sel[0]] has the LEAST bf16 compounding,
    # so it's the cleanest test that the (hidden+residual) capture point is structurally correct.
    D = e_hlast.shape[-1]; m = e_hsel.shape[-1] // D
    for j in range(m):
        mx, mn, r = rel(e_hsel[..., j * D:(j + 1) * D], s_hsel[..., j * D:(j + 1) * D])
        tag = "  <== structural (earliest)" if j == 0 else ""
        print(f"  h_sel[hs{sel[j]:>2}]  max|Δ|={mx:.4f} mean|Δ|={mn:.5f} rel={r:.4f}{tag}")
    mx, mn, r = rel(e_hlast, s_hlast); print(f"  h_last       max|Δ|={mx:.4f} mean|Δ|={mn:.5f} rel={r:.4f}")
    argm = float("nan")
    if d["logits"] is not None:
        s_log = d["logits"].to(device)
        mx, mn, r = rel(e_logits, s_log)
        argm = (e_logits.argmax(-1) == s_log.argmax(-1)).float().mean().item()
        print(f"  logits       max|Δ|={mx:.4f} mean|Δ|={mn:.5f} rel={r:.4f}  ARGMAX_MATCH={argm:.4f}")

    _, _, r0 = rel(e_hsel[..., :D], s_hsel[..., :D])
    ok = (argm == argm and argm > 0.95) or (r0 < 0.25)
    print(f"\nVERDICT: {'STRUCTURAL PASS' if ok else 'STRUCTURAL FAIL'} — "
          + ("logits argmax agree and/or earliest h_sel rel-Δ is rounding-scale => same computation, build G2."
             if ok else
             "logits argmax disagree AND earliest h_sel rel-Δ large => wrong capture point / model mismatch."))
    print("  (bf16 across two implementations: judge by ARGMAX_MATCH (should be ~1.0) + hs%d rel-Δ (rounding),"
          " NOT absolute Δ — 20 layers of bf16 compound. The definitive test remains end-to-end drafter accuracy.)"
          % sel[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["sglang", "eager"])
    p.add_argument("--heavy_path", required=True)
    p.add_argument("--drafter_path", default=None, help="eager mode only")
    p.add_argument("--dump", default="/tmp/g1_hsel.pt")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--exact_moe", action="store_true")
    args = p.parse_args()
    if args.mode == "sglang":
        run_sglang(args)
    else:
        assert args.drafter_path, "--mode eager needs --drafter_path"
        run_eager(args)


if __name__ == "__main__":
    main()
