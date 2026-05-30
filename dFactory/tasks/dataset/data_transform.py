import torch
from typing import  Any, Dict, List, Optional, Sequence, Union, Tuple



def process_mdm_sft_example(
    example: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
    text_keys: Union[str, List[str]] = "messages",
    noise_range: Tuple[float, float] = (0.3, 0.8),
    mask_token_id: int = 156895,
    source_name: Optional[str] = None,
    # T3-D ADDED: optional step-based mask ramp. When set, sigma ramps linearly from
    # noise_range[0] (at progress=0) to noise_range[1] (at progress=1). Without it,
    # sigma is uniform-sampled in [noise_range[0], noise_range[1]] per sample (DMax default).
    progress_state: Optional[Any] = None,
    # T3-D v2 ADDED: stochastic gate width on sigma (0.0 disables). Per-sample sigma is
    # drawn uniformly in [ramp_center - gate, ramp_center + gate], clipped to [0, 1].
    sigma_gate: float = 0.0,
) -> List[Dict[str, "torch.Tensor"]]:
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
    
    examples = []
    input_ids, prompt_length = apply_chat_template_mdm(messages=messages, tokenizer=tokenizer, max_length=max_seq_len)

    labels = input_ids.clone()
    labels[:prompt_length] = -100

    maskable_mask = torch.arange(max_seq_len) >= prompt_length
    
    noisy_input_ids = sft_noise_transition(
        input_ids.clone(),
        noise_range=noise_range,
        maskable_mask=maskable_mask,
        mask_token_id=mask_token_id,
        progress_state=progress_state,
        sigma_gate=sigma_gate,
    )

    loss_mask = noisy_input_ids == mask_token_id
    # T3-D FIX (2026-05-31): the SFT-label leak. Previously commented out (DMax inherited
    # the bug from its own SFT path; their tokenized path at line 122 has it active). With
    # same_token_labels=True (v6e config), keeping labels at unmasked response positions
    # lets the model satisfy CE via identity copy through the tied lm_head/embedding -- no
    # delta_head pressure, training loss collapses to ~0.098 without real learning.
    # Restoring the LLaDA paper's masked-token-only objective: loss only at MASK positions.
    labels[~loss_mask] = -100

    eos_id = tokenizer.pad_token_id  # endtoken id

    # 找到末尾连续 eos 的起始位置（trailing EOS run）
    # 例如: [...., x, eos, eos, eos] -> run_start 指向第一个 eos
    not_eos = (input_ids != eos_id)

    if torch.any(not_eos):
        # last index where token != eos
        last_not_eos = torch.nonzero(not_eos, as_tuple=False)[-1].item()
        run_start = last_not_eos + 1  # trailing eos run starts here
    else:
        # 极端情况：全是 eos（基本不该发生），那就只保留第0个有loss
        run_start = 0

    # 仅当确实存在 trailing eos run（run_start < max_seq_len）时处理
    if run_start < input_ids.numel():
        # trailing eos run 的前32个 eos 可以有 loss，其后的 eos 都设为 -100
        # 同时确保不动 prompt 之前的 -100 设定
        start = max(run_start + 32, prompt_length)
        if start < input_ids.numel():
            labels[start:] = -100
        
  
    examples.append({
        "input_ids": input_ids,
        "noisy_input_ids": noisy_input_ids,
        "attention_mask": torch.ones_like(input_ids), # 使用torch.ones_like更简洁
        "labels": labels,
        "flag": torch.tensor(example["flag"])
    })

 
    return examples


def process_mdm_tokenized_example(
    example: Dict[str, List[int]],
    max_seq_len: int, 
    text_keys: Union[str, List[str]] = "input_ids",
    noise_range: Tuple[float, float] = (0.3, 0.8),
    mask_token_id: int = 156895,
    source_name: Optional[str] = None,
) -> List[Dict[str, "torch.Tensor"]]:
    examples = []
    if isinstance(text_keys, str):
        input_ids = example[text_keys]
    elif isinstance(text_keys, list):
        for text_key in text_keys:
            if text_key in example:
                input_ids = example[text_key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    prompt_length = example['prompt_lengths']

    input_ids = torch.tensor(input_ids)
    labels = input_ids.clone()
    labels[:prompt_length] = -100

    maskable_mask = torch.arange(max_seq_len) >= prompt_length

    noisy_input_ids = sft_noise_transition(
        input_ids.clone(), 
        noise_range=noise_range, 
        maskable_mask=maskable_mask, 
        mask_token_id=mask_token_id
    )

    loss_mask = noisy_input_ids == mask_token_id
    labels[~loss_mask] = -100

    examples.append({
        "input_ids": input_ids,
        "noisy_input_ids": noisy_input_ids,
        "attention_mask": torch.ones_like(input_ids), # 使用torch.ones_like更简洁
        "labels": labels,
    })
    
    return examples



def sft_noise_transition(x_0, noise_range, maskable_mask, mask_token_id, progress_state=None, sigma_gate=0.0):
    """
    Performs a noise transition by masking tokens.

    Args:
        x_0 (torch.Tensor): The input sequence (batch_size, seq_len).
        noise_range (tuple): A tuple (min, max) for the noise range, from which the masking
            ratio sigma is sampled (per-sample uniform) OR used as the ramp endpoints
            (when progress_state is provided).
        maskable_mask (torch.Tensor): A boolean mask indicating which positions are allowed
            to be masked (batch_size, seq_len).
        mask_token_id (int): The ID of the mask token.
        progress_state (Optional[multiprocessing.Value or float]): If provided, the masking
            ratio is computed deterministically as
                sigma = noise_range[0] + (noise_range[1] - noise_range[0]) * progress
            where `progress` is read from progress_state (`.value` if Value, else float).
            The training script writes the current step/total_steps fraction here so each
            worker sees a smoothly-ramped sigma. Without it, behaviour is unchanged
            (per-sample uniform sampling).
        sigma_gate (float): T3-D v2 stochastic gate width on sigma. When > 0 AND
            progress_state is provided, per-sample sigma is drawn uniformly from
            [center - sigma_gate, center + sigma_gate] around the ramp center,
            clipped to [0, 1]. 0.0 = no gate (deterministic ramp).

    Returns:
        torch.Tensor: The sequence after masking.
    """
    if progress_state is not None:
        # Lock-free read; mp.Value.value is a single double, atomic enough for our smooth
        # ramp. Workers may see a slightly stale value (prefetch latency), which is fine.
        if hasattr(progress_state, "value"):
            progress = float(progress_state.value)
        else:
            progress = float(progress_state)
        progress = max(0.0, min(1.0, progress))
        center = noise_range[0] + (noise_range[1] - noise_range[0]) * progress
        if sigma_gate > 0.0:
            # T3-D v2 stochastic gate: per-sample sigma in [center - gate, center + gate].
            sigma = float(torch.empty(1).uniform_(center - sigma_gate, center + sigma_gate).item())
            sigma = max(0.0, min(1.0, sigma))
        else:
            sigma = center
    else:
        t_tensor = torch.rand(1) * (noise_range[1] - noise_range[0]) + noise_range[0]
        sigma = t_tensor.item()
    # move_chance = 1 - (-sigma).exp()
    move_chance = sigma
    move_indices = (torch.rand(*x_0.shape) < move_chance) & maskable_mask
    x_t = torch.where(move_indices, mask_token_id, x_0)
    return x_t






def apply_chat_template_mdm(messages, tokenizer, max_length):
    inputs_str = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_str = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

    prompt_ids_unpadded = tokenizer(prompt_str, add_special_tokens=False)['input_ids']
    prompt_length = len(prompt_ids_unpadded)

    tokenized_input = tokenizer(
        inputs_str,
        return_tensors="pt",
        truncation=True, 
        max_length=max_length, 
        padding="max_length",
        add_special_tokens=False
    ).input_ids.squeeze(0)

    return tokenized_input, prompt_length