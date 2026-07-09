# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
"""DBet TRAIN-V2 trainer — the G/A dual-rollout + mask-route + on-policy recipe (experiment.md §train-v2).

An INDEPENDENT entrypoint that reuses train_dbet.py's VeOmni machinery (dist init, dataloader, parallelize,
optimizer/scheduler, checkpointing, metrics/eval loop) verbatim instead of forking 800 lines: it extends the
training arguments with the v2 knobs and swaps the core step for `dbet_v2_core.dbet_v2_train_step` (drop-in
signature) before delegating to train_dbet.main(). Everything else — data transform, dual-stream assembly,
FSDP, hf export — is identical to v1.

Config: configs/sft/dbet_v2_1gpu.yaml. Two v1 settings it MUST override:
  - data.noise_range_low/high = 1.0/1.0  -> the noisy answer region arrives ALL-MASK (the G/A rollouts
    generate the states themselves; a partial golden reveal would contaminate the reckless stage);
  - the v1 knobs heavy_commit_threshold / align_to_2nd_pass are inert here (the v2 step samples its own
    thresholds and always targets the verifier), kept only for the shared eval probe.

Launch:
  PYTHONPATH=$(pwd)/VeOmni:$(pwd):$PYTHONPATH bash train.sh tasks/train_dbet_v2.py configs/sft/dbet_v2_1gpu.yaml
"""

from dataclasses import dataclass, field

import train_dbet
from train_dbet import LLaDA2DataArguments, LLaDA2ModelArguments, LLaDA2TrainingArguments
from dbet_v2_core import dbet_v2_train_step


@dataclass
class V2TrainingArguments(LLaDA2TrainingArguments):
    # --- step 1: the careful reference (G rollout) ---
    g_passes: int = field(default=5, metadata={"help": "V2: heavy passes for the careful reference rollout."})
    g_threshold: float = field(default=0.9, metadata={"help": "V2: commit threshold of the careful rollout."})
    # --- step 2: the reckless stage (A pass) ---
    a_thresholds: str = field(default="0.3,0.5,0.7", metadata={
        "help": "V2: comma-separated aggressive commit thresholds; one is sampled per micro-batch."})
    # --- step 5: route-M weights (route H reuses align_ce_weight / align_l1_weight / golden_ce_weight) ---
    m_ce_p_weight: float = field(default=0.3, metadata={
        "help": "V2 route M: CE weight toward the careful reference p (trust-window gated)."})
    m_l1_weight: float = field(default=0.7, metadata={"help": "V2 route M: L1 weight toward the verifier p'."})
    # --- step 6: the on-policy sendback (deployment-matched drafter commit) ---
    sendback_extend_threshold: float = field(default=0.8, metadata={"help": "V2: drafter EXTEND conf gate."})
    sendback_fix_threshold: float = field(default=0.9, metadata={"help": "V2: drafter FIX conf gate."})
    sendback_top_k: int = field(default=2, metadata={"help": "V2: soft-embed top-k for drafter commits."})
    sendback_tau: float = field(default=1.0, metadata={"help": "V2: soft-embed temperature for drafter commits."})


@dataclass
class V2Arguments:
    model: "LLaDA2ModelArguments" = field(default_factory=LLaDA2ModelArguments)
    data: "LLaDA2DataArguments" = field(default_factory=LLaDA2DataArguments)
    train: "V2TrainingArguments" = field(default_factory=V2TrainingArguments)


def main():
    train_dbet.Arguments = V2Arguments               # parse_args picks up the v2 knobs
    train_dbet.dbet_train_step = dbet_v2_train_step  # the training loop calls the v2 core step
    train_dbet.main()


if __name__ == "__main__":
    main()
