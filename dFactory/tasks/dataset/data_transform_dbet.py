"""DBet data transform — DMax's `data_transform.py` with the noising swapped from random masking to
**left-to-right per-block reveal** (and `dbet_data.py` merged in: the block-diffusion attention mask).

Mirrors `data_transform.process_mdm_sft_example` / `process_mdm_tokenized_example` so it drops into the
OPUT training framework (`train_dbet.py` ← `train_llada2_bd_oput.py`); only the noise transition differs:

  random `sft_noise_transition` (Bernoulli mask)  ->  `block_left_to_right_reveal`:
    per grid block, sample a mask ratio σ (same `noise_range` / progress-ramp / sigma-gate machinery as DMax),
    reveal the LEFTMOST round((1-σ)·#maskable) tokens of the block and mask the rest. An 8-token block at
    σ=0.75 (reveal 0.25) keeps its 2 leftmost maskable tokens. This matches inference (DMax decode_uniform
    commits a left-to-right prefix), so train ≈ infer.

The dual-stream `[noisy|clean]` + the block-diffusion mask (M_BD | M_OBC | M_BC) are built in `train_dbet.py`
(data-independent, once per run) — `build_block_diffusion_attn_mask` here is the shared source of that mask.
"""

import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union  # noqa: F401

MASK_TOKEN_ID = 156895


# ============================================================================
#                       noising — left-to-right per-block reveal
# ============================================================================
def _sample_sigma(noise_range, progress_state=None, sigma_gate=0.0):
    """Mask ratio σ — identical machinery to data_transform.sft_noise_transition (per-sample uniform, or a
    progress-ramped center with an optional stochastic gate)."""
    if progress_state is not None:
        progress = float(progress_state.value) if hasattr(progress_state, "value") else float(progress_state)
        progress = max(0.0, min(1.0, progress))
        center = noise_range[0] + (noise_range[1] - noise_range[0]) * progress
        if sigma_gate > 0.0:
            sigma = float(torch.empty(1).uniform_(center - sigma_gate, center + sigma_gate).item())
            return max(0.0, min(1.0, sigma))
        return center
    return float(torch.rand(1) * (noise_range[1] - noise_range[0]) + noise_range[0])


def block_left_to_right_reveal(x_0, noise_range, maskable_mask, mask_token_id, block_size,
                               progress_state=None, sigma_gate=0.0):
    """Noise transition by LEFT-TO-RIGHT per-block reveal (the DBet replacement for random `sft_noise_transition`).
    Grid-aligned blocks from position 0; per block, keep the leftmost round((1-σ)·#maskable) maskable tokens and
    mask the rest. Drop-in signature (adds `block_size`). x_0 [L] -> x_t [L]."""
    sigma = _sample_sigma(noise_range, progress_state, sigma_gate)
    reveal_ratio = max(0.0, min(1.0, 1.0 - sigma))
    L = x_0.shape[0]
    x_t = x_0.clone()
    for bs in range(0, L, block_size):
        be = min(bs + block_size, L)
        maskable = [p for p in range(bs, be) if bool(maskable_mask[p])]
        if not maskable:
            continue
        keep = int(round(reveal_ratio * len(maskable)))
        for p in maskable[keep:]:                      # mask everything past the revealed left prefix
            x_t[p] = mask_token_id
    return x_t


# ============================================================================
#                       example builders (mirror data_transform.py)
# ============================================================================
def process_mdm_sft_example(
    example: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
    block_size: int,
    text_keys: Union[str, List[str]] = "messages",
    noise_range: Tuple[float, float] = (0.3, 0.8),
    mask_token_id: int = MASK_TOKEN_ID,
    source_name: Optional[str] = None,
    progress_state: Optional[Any] = None,
    sigma_gate: float = 0.0,
) -> List[Dict[str, "torch.Tensor"]]:
    """messages -> chat-template tokens -> (input_ids clean, noisy_input_ids left-to-right reveal, labels,
    attention_mask=ones, flag). Same as data_transform.process_mdm_sft_example except the noise transition."""
    if isinstance(text_keys, str):
        messages = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                messages = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    input_ids, prompt_length = apply_chat_template_mdm(messages=messages, tokenizer=tokenizer, max_length=max_seq_len)
    labels = input_ids.clone()
    labels[:prompt_length] = -100
    maskable_mask = torch.arange(max_seq_len) >= prompt_length

    noisy_input_ids = block_left_to_right_reveal(
        input_ids.clone(), noise_range, maskable_mask, mask_token_id, block_size,
        progress_state=progress_state, sigma_gate=sigma_gate,
    )

    loss_mask = noisy_input_ids == mask_token_id
    labels[~loss_mask] = -100                           # loss only at MASK positions (LLaDA masked-token objective)

    eos_id = tokenizer.pad_token_id
    not_eos = (input_ids != eos_id)
    if torch.any(not_eos):
        run_start = int(torch.nonzero(not_eos, as_tuple=False)[-1].item()) + 1
    else:
        run_start = 0
    if run_start < input_ids.numel():                  # trailing-EOS run: keep ~32 with loss, rest -100
        start = max(run_start + 32, prompt_length)
        if start < input_ids.numel():
            labels[start:] = -100

    return [{
        "input_ids": input_ids,
        "noisy_input_ids": noisy_input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "flag": torch.tensor(example.get("flag", False)),   # DBet ignores flag; optional so flag-less data is OK
    }]


def process_mdm_tokenized_example(
    example: Dict[str, List[int]],
    max_seq_len: int,
    block_size: int,
    text_keys: Union[str, List[str]] = "input_ids",
    noise_range: Tuple[float, float] = (0.3, 0.8),
    mask_token_id: int = MASK_TOKEN_ID,
    source_name: Optional[str] = None,
    progress_state: Optional[Any] = None,
    sigma_gate: float = 0.0,
) -> List[Dict[str, "torch.Tensor"]]:
    """Pre-tokenized variant (input_ids + prompt_lengths). Same output schema (no flag), left-to-right reveal."""
    if isinstance(text_keys, str):
        input_ids = example[text_keys]
    elif isinstance(text_keys, list):
        for k in text_keys:
            if k in example:
                input_ids = example[k]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    prompt_length = example["prompt_lengths"]
    input_ids = torch.tensor(input_ids)
    labels = input_ids.clone()
    labels[:prompt_length] = -100
    maskable_mask = torch.arange(max_seq_len) >= prompt_length

    noisy_input_ids = block_left_to_right_reveal(
        input_ids.clone(), noise_range, maskable_mask, mask_token_id, block_size,
        progress_state=progress_state, sigma_gate=sigma_gate,
    )
    loss_mask = noisy_input_ids == mask_token_id
    labels[~loss_mask] = -100

    return [{
        "input_ids": input_ids,
        "noisy_input_ids": noisy_input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
    }]


def apply_chat_template_mdm(messages, tokenizer, max_length):
    """Verbatim from data_transform.apply_chat_template_mdm: chat-template -> padded token ids + prompt_length."""
    inputs_str = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_str = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    prompt_length = len(tokenizer(prompt_str, add_special_tokens=False)["input_ids"])
    tokenized_input = tokenizer(
        inputs_str, return_tensors="pt", truncation=True, max_length=max_length,
        padding="max_length", add_special_tokens=False,
    ).input_ids.squeeze(0)
    return tokenized_input, prompt_length


# ============================================================================
#       dual-stream block-diffusion attention mask (merged from dbet_data.py)
# ============================================================================
def block_diffusion_keep_flag(q_idx, kv_idx, block_size, n):
    """Boolean keep-mask for the dual stream `[noisy(0..n-1) | clean(n..2n-1)]` — M_BD | M_OBC | M_BC, verbatim
    logic from train_llada2_bd.py:block_diffusion_mask:
      - noisy block i attends its OWN noisy block (M_BD) + clean blocks STRICTLY < i (M_OBC; == index = leakage);
      - clean block i attends clean blocks <= i (M_BC); clean never attends noisy.
    q_idx/kv_idx: broadcastable index tensors over [0, 2n)."""
    x0_q, x0_kv = q_idx >= n, kv_idx >= n
    bq = torch.where(x0_q, (q_idx - n) // block_size, q_idx // block_size)
    bkv = torch.where(x0_kv, (kv_idx - n) // block_size, kv_idx // block_size)
    m_bd = (bq == bkv) & (x0_q == x0_kv)
    m_obc = (bq > bkv) & x0_kv & (~x0_q)
    m_bc = (bq >= bkv) & x0_kv & x0_q
    return m_bd | m_obc | m_bc


def build_block_diffusion_attn_mask(n, block_size, dtype, device):
    """4D additive mask [1,1,2n,2n] for the dual stream (disallowed -> -inf). Build once per (n, block_size) and
    `.expand(B,-1,-1,-1)` per batch (data-independent), like train_llada2_bd.py:377-381/434."""
    idx = torch.arange(2 * n, device=device)
    keep = block_diffusion_keep_flag(idx[:, None], idx[None, :], block_size, n)
    mask = torch.zeros(1, 1, 2 * n, 2 * n, dtype=dtype, device=device)
    mask.masked_fill_(~keep, float("-inf"))
    return mask


def block_diffusion_mask_mod(block_size, n):
    """flex_attention mask_mod factory: `(b, h, q_idx, kv_idx) -> bool` (H200-fast path)."""
    def mask_mod(b, h, q_idx, kv_idx):
        return block_diffusion_keep_flag(q_idx, kv_idx, block_size, n)
    return mask_mod


if __name__ == "__main__":  # quick self-test (run where torch is installed)
    L, B, P = 24, 8, 8
    clean = torch.arange(100, 100 + L)
    maskable = torch.arange(L) >= P
    noisy = block_left_to_right_reveal(clean, (0.75, 0.75), maskable, MASK_TOKEN_ID, B)  # σ=0.75 -> reveal 0.25
    assert (noisy[:P] == clean[:P]).all()
    for bs in range(P, L, B):
        assert (noisy[bs:bs + 2] == clean[bs:bs + 2]).all() and (noisy[bs + 2:bs + B] == MASK_TOKEN_ID).all()
    m = build_block_diffusion_attn_mask(L, B, torch.float32, "cpu")[0, 0]
    keep = torch.isfinite(m)
    assert not keep[8, L + 8] and keep[8, L + 0] and not keep[L + 8, L + 16] and not keep[L + 8, 0]
    print("data_transform_dbet selftest OK")
