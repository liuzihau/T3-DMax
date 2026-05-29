# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
#
# lm-evaluation-harness wrapper for T3-D. Mirrors the role of DMax's
# `eval_dinfer.py` / `eval_dinfer_sglang.py` but stays single-process and
# single-GPU -- no vllm/sglang, no CUDA graphs, no tensor parallel.
#
# Usage (matches DMax's pattern; see eval_t3d_mini.sh):
#   python eval_dinfer_t3d.py \
#     --tasks gsm8k_llada_mini \
#     --confirm_run_unsafe_code \
#     --model t3d_eval \
#     --model_args model_path=...,gen_length=...,block_length=32,threshold=0.9 \
#     --output_path ... \
#     --include_path /path/to/dInfer/evaluations/tasks \
#     --apply_chat_template
#
# Task YAMLs (e.g. `tasks/gsm8k/gsm8k-llada-mini.yaml`) are reused verbatim from
# DMax. lm-eval-harness handles:
#   - dataset loading (HuggingFace `gsm8k` / `main` test split)
#   - few-shot prompt construction
#   - chat-template application
#   - per-task regex filters + exact_match scoring

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import torch
from tqdm import tqdm

# lm-eval-harness imports (the eval_dinfer.py pattern).
from lm_eval import evaluator
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

# Local: import the T3-D inference shim + decoding loop.
_HERE = os.path.dirname(os.path.abspath(__file__))
# .../T3-DMax/dInfer/evaluations -> .../T3-DMax/dInfer/python
_DINFER_PYTHON = os.path.abspath(os.path.join(_HERE, "..", "python"))
if _DINFER_PYTHON not in sys.path:
    sys.path.insert(0, _DINFER_PYTHON)

from dinfer.model.modeling_think_talk_t3d import ThinkTalkT3DInference  # noqa: E402
from dinfer.decoding.generate_t3d import generate_t3d, T3DGenerateStats  # noqa: E402


# ============================================================================
#                              lm_eval LM subclass
# ============================================================================

@register_model("t3d_eval")
class T3DEvalHarness(LM):
    """lm-eval-harness model that runs T3-D's block-diffusion decoding loop
    for each `generate_until` request. Single-GPU, single-process."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str = None,
        device: str = "cuda",
        batch_size: int = 1,
        max_length: int = 4096,
        gen_length: int = 512,
        block_length: int = 32,
        threshold: float = 0.9,
        max_iter_per_block: int = 32,
        early_stop: bool = True,
        save_dir: str = None,
        save_samples: bool = False,
        show_speed: bool = True,
        **kwargs,
    ):
        super().__init__()
        # Coerce numeric kwargs (lm-eval passes them as strings from --model_args).
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.gen_length = int(gen_length)
        self.block_length = int(block_length)
        self.threshold = float(threshold)
        self.max_iter_per_block = int(max_iter_per_block)
        self.early_stop = bool(int(early_stop)) if isinstance(early_stop, str) else bool(early_stop)
        self.save_dir = save_dir
        self.save_samples = bool(int(save_samples)) if isinstance(save_samples, str) else bool(save_samples)
        self.show_speed = bool(int(show_speed)) if isinstance(show_speed, str) else bool(show_speed)
        self.device = device

        # Single-process: rank 0, world size 1.
        self._rank = 0
        self._world_size = 1

        # Load model + tokenizer.
        self.inference = ThinkTalkT3DInference(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            device=device,
            dtype=torch.bfloat16,
        )
        self.tokenizer = self.inference.tokenizer

        # Default model knobs (overrideable per-task via YAML generation_kwargs).
        self.kwargs = kwargs

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    # --------------------------------------------------------- lm_eval bookkeeping
    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def eot_token_id(self):
        return self.inference.eos_id

    @property
    def max_gen_toks(self):
        return self.gen_length

    @property
    def max_length_property(self):
        return self.max_length

    # --------------------------------------------------------- generate_until
    def generate_until(self, requests) -> List[str]:
        """Iterate over lm-eval requests and decode each. Each request's first
        positional arg is the (chat-templated) prompt string."""
        save_path = None
        if self.save_dir is not None:
            save_path = os.path.join(self.save_dir, f"rank_{self.rank}.jsonl")
            print(f"[T3-D eval] writing samples to {save_path}")

        answers: List[str] = []
        total_think = 0
        total_talk = 0
        total_tokens = 0
        t0 = time.time()

        write_handle = open(save_path, "w", encoding="utf-8") if save_path else None
        try:
            for i, req in enumerate(tqdm(requests, desc="T3-D decode")):
                prompt_text = req.args[0]
                input_ids = self.tokenizer(
                    prompt_text, return_tensors="pt", add_special_tokens=False,
                ).input_ids.to(self.device)
                if input_ids.shape[1] > self.max_length - self.gen_length:
                    # Hard truncate from the left to fit in the model's seq budget.
                    input_ids = input_ids[:, -(self.max_length - self.gen_length):]

                t_block = time.time()
                full_ids, stats = generate_t3d(
                    inference=self.inference,
                    prompt_ids=input_ids,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    threshold=self.threshold,
                    max_iter_per_block=self.max_iter_per_block,
                    early_stop=self.early_stop,
                )
                dt = time.time() - t_block

                response_ids = full_ids[0, input_ids.shape[1]:]
                answer_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                answers.append(answer_text)

                total_think += stats.think_forwards
                total_talk += stats.talk_forwards
                gen_tokens = response_ids.shape[0]
                total_tokens += gen_tokens

                if self.show_speed and self.rank == 0:
                    tps = gen_tokens / max(dt, 1e-6)
                    tqdm.write(
                        f"[T3-D iter={i}] think={stats.think_forwards} "
                        f"talk={stats.talk_forwards} tps={tps:.1f}"
                    )

                if write_handle is not None:
                    write_handle.write(json.dumps({
                        "index": i,
                        "prompt": prompt_text,
                        "answer": answer_text,
                        "think_forwards": stats.think_forwards,
                        "talk_forwards": stats.talk_forwards,
                        "gen_tokens": gen_tokens,
                        "seconds": dt,
                    }, ensure_ascii=False) + "\n")
                    write_handle.flush()
        finally:
            if write_handle is not None:
                write_handle.close()

        elapsed = time.time() - t0
        if self.show_speed and self.rank == 0:
            tpf_think = total_tokens / max(total_think, 1)
            tpf_talk = total_tokens / max(total_talk, 1)
            print(
                f"[T3-D] total: think_forwards={total_think} talk_forwards={total_talk} "
                f"tokens={total_tokens} time={elapsed:.1f}s "
                f"TPF_think={tpf_think:.2f} TPF_talk={tpf_talk:.2f}"
            )

        if save_path is not None and not self.save_samples:
            os.remove(save_path)

        return answers

    # --------------------------------------------------------- not implemented
    def loglikelihood(self, requests):
        raise NotImplementedError(
            "T3-D doesn't yet support log-likelihood scoring; only generate_until tasks."
        )

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError


# ============================================================================
#                                   __main__
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(1234)
    cli_evaluate()
