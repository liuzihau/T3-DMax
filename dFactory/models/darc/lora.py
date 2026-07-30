# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# Minimal, TOGGLEABLE LoRA for the frozen DMax decoder layers. A trainable low-rank delta on a frozen Linear;
# B is zero-init so the layer starts == the frozen base. `.enabled` lets us run the layer WITHOUT the delta
# (the honest frozen baseline for the no-tap metrics) and WITH it (the DARC replay of the injected h_ar).

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """out = base(x) + enabled * scale * (x Aᵀ) Bᵀ ;  B zero-init -> delta == 0 at init.  base frozen."""

    def __init__(self, base: nn.Linear, r=16, alpha=32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / r
        self.enabled = True
        dev = base.weight.device                                     # fp32 A/B on the base's device (base stays
        self.A = nn.Parameter(torch.zeros(r, base.in_features, device=dev))         # bf16 & untouched -- never
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))        # .to() the whole module)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))             # B stays zero -> delta == 0 at init

    def forward(self, x):
        out = self.base(x)
        if self.enabled:
            out = out + self.scale * F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype))
        return out


def add_lora(model, layer_indices, r=16, alpha=32, targets=("query_key_value", "dense")):
    """Wrap the named Linear(s) in model.model.layers[i].attention with LoRALinear. Returns the LoRALinear
    modules (for the optimizer param-group + the enable/disable toggle)."""
    mods = []
    for i in layer_indices:
        attn = model.model.layers[i].attention
        for name in targets:
            lin = getattr(attn, name)
            if not isinstance(lin, nn.Linear):
                raise TypeError(f"layers[{i}].attention.{name} is {type(lin).__name__}, expected nn.Linear")
            lora = LoRALinear(lin, r, alpha)                         # A/B fp32 on-device; base (lin) stays bf16
            setattr(attn, name, lora)
            mods.append(lora)
    return mods


def set_lora_enabled(mods, enabled):
    for m in mods:
        m.enabled = enabled


def lora_parameters(mods):
    ps = []
    for m in mods:
        ps += [m.A, m.B]
    return ps


def lora_state_dict(mods):
    """Only the trainable A/B deltas (the frozen base lives in the model checkpoint, not here)."""
    return [{"A": m.A.detach().cpu(), "B": m.B.detach().cpu(), "scale": m.scale} for m in mods]


def load_lora_state(mods, state):
    for m, s in zip(mods, state):
        with torch.no_grad():
            m.A.copy_(s["A"].to(device=m.A.device, dtype=m.A.dtype))
            m.B.copy_(s["B"].to(device=m.B.device, dtype=m.B.dtype))
    return min(len(mods), len(state))


if __name__ == "__main__":
    torch.manual_seed(0)
    base = nn.Linear(16, 24).to(torch.bfloat16)                     # mimic the frozen bf16 backbone Linear
    base_w = base.weight.detach().clone()
    lin = LoRALinear(base, r=4, alpha=8)
    assert lin.base.weight.dtype == torch.bfloat16, "base dtype corrupted (must stay bf16)"
    assert lin.A.dtype == torch.float32 and lin.B.dtype == torch.float32, "A/B must be fp32 master"
    x = torch.randn(2, 5, 16, dtype=torch.bfloat16)

    # at init: B=0 -> delta 0 -> enabled output == base output
    lin.enabled = True
    assert torch.allclose(lin(x), base(x)), "delta != 0 at init (B not zero?)"

    # a non-trivial delta only shows when enabled
    with torch.no_grad():
        lin.B.add_(0.1)
    lin.enabled = False
    assert torch.allclose(lin(x), base(x)), "OFF must equal the frozen base"
    lin.enabled = True
    assert not torch.allclose(lin(x), base(x)), "ON must differ from base once B!=0"

    # grad reaches A/B, never the frozen base
    lin.zero_grad(set_to_none=True)
    lin(x).float().pow(2).mean().backward()
    assert lin.A.grad is not None and lin.B.grad is not None, "A/B got no grad"
    assert base.weight.grad is None, "frozen base got grad"
    assert torch.equal(lin.base.weight.detach(), base_w), "base weight changed"

    # toggle helper + save/load roundtrip
    set_lora_enabled([lin], False); assert lin.enabled is False
    st = lora_state_dict([lin])
    with torch.no_grad():
        lin.A.zero_(); lin.B.zero_()
    load_lora_state([lin], st)
    assert torch.allclose(lin.B, st[0]["B"].to(lin.B.dtype)), "load_lora_state roundtrip failed"
    print("lora self-test PASSED  (base stays bf16/frozen, A/B fp32 trainable, toggle+save/load OK)")
