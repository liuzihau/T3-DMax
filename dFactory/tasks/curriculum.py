"""T3-D v2 curriculum sampler.

Extracted from train_t3_dmax_bd_oput.py so it can be unit-tested without
pulling in VeOmni / FSDP / etc. The trainer imports from here; tests/ does
the same. Pure: takes floats/ints, returns a dict.
"""

import torch


def sample_curriculum(
    progress,
    noise_range_low,
    noise_range_high,
    sigma_gate,
    rollout_low,
    rollout_high,
    rollout_gate,
    n_min,
    n_max,
    n_gate,
):
    """Per-step T3-D v2 curriculum sampler. Returns (centers + samples) for the
    three-dim ramp (sigma, rollout_ratio, N). Each dim independently sampled
    with its stochastic gate around its ramp center.

    Args:
      progress:          float in [0, 1] -- current_step / total_steps
      noise_range_low:   sigma ramp endpoint at progress=0
      noise_range_high:  sigma ramp endpoint at progress=1
      sigma_gate:        stochastic gate width on sigma (0.0 disables)
      rollout_low:       rollout_ratio ramp endpoint at progress=0
      rollout_high:      rollout_ratio ramp endpoint at progress=1
      rollout_gate:      stochastic gate width on rollout_ratio (0.0 disables)
      n_min:             curriculum start for N (iter count)
      n_max:             curriculum end for N
      n_gate:            integer gate width on N per step (0 disables)

    Returns:
      dict with sigma_center, sigma_sample, rollout_center, rollout_sample,
      n_center, n_sample
    """
    progress = max(0.0, min(1.0, float(progress)))

    # Sigma: ramp low -> high. Optional ±gate.
    sigma_center = noise_range_low + (noise_range_high - noise_range_low) * progress
    sigma_gate = float(sigma_gate)
    if sigma_gate > 0.0:
        sigma_sample = float(torch.empty(1).uniform_(
            sigma_center - sigma_gate, sigma_center + sigma_gate
        ).item())
        sigma_sample = max(0.0, min(1.0, sigma_sample))
    else:
        sigma_sample = sigma_center

    # Rollout ratio: ramp low -> high. Optional ±gate.
    rollout_center = rollout_low + (rollout_high - rollout_low) * progress
    rollout_gate = float(rollout_gate)
    if rollout_gate > 0.0:
        rollout_sample = float(torch.empty(1).uniform_(
            rollout_center - rollout_gate, rollout_center + rollout_gate
        ).item())
        rollout_sample = max(0.0, min(1.0, rollout_sample))
    else:
        rollout_sample = rollout_center

    # N iterations: ramp n_min -> n_max. ±n_gate (integer).
    n_min = max(1, int(n_min))
    n_max = max(n_min, int(n_max))
    n_center = n_min + (n_max - n_min) * progress
    n_gate = int(n_gate)
    if n_gate > 0:
        c = int(round(n_center))
        lo, hi = c - n_gate, c + n_gate
        n_sample = int(torch.randint(lo, hi + 1, (1,)).item())
    else:
        n_sample = int(round(n_center))
    n_sample = max(1, min(7, n_sample))

    return {
        "sigma_center": sigma_center,
        "sigma_sample": sigma_sample,
        "rollout_center": rollout_center,
        "rollout_sample": rollout_sample,
        "n_center": n_center,
        "n_sample": n_sample,
    }
