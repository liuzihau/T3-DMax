"""Unit tests for T3-D v2 curriculum sampler.

Verifies that the sigma / rollout_ratio / N samplers respect the stochastic
gate widths and clip bounds across the progress range. Uses the standalone
tasks.curriculum module so this test doesn't drag in VeOmni / FSDP.

Run from the T3-DMax repo root:
    PYTHONPATH=dFactory:$PYTHONPATH pytest tests/test_curriculum_sampler.py -v
"""

import importlib.util
import os

import pytest

# Skip the whole module if torch isn't installed (e.g., on a dev box without
# the training env). The trainer's runtime always has torch.
pytest.importorskip("torch")


def _load_sampler():
    """Import sample_curriculum without requiring tasks/__init__.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "dFactory", "tasks", "curriculum.py"))
    spec = importlib.util.spec_from_file_location("curriculum_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sample_curriculum


@pytest.fixture(scope="module")
def sample_curriculum():
    return _load_sampler()


# v2 redesign endpoints from training_redesign_plan.md §1.3.
V2_DEFAULTS = dict(
    noise_range_low=0.50, noise_range_high=0.90,
    sigma_gate=0.10,
    rollout_low=0.20, rollout_high=0.60,
    rollout_gate=0.10,
    n_min=2, n_max=5, n_gate=1,
)


@pytest.mark.parametrize("progress", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_sigma_center_is_linear(sample_curriculum, progress):
    c = sample_curriculum(progress, **V2_DEFAULTS)
    expected_center = 0.50 + 0.40 * progress
    assert abs(c["sigma_center"] - expected_center) < 1e-9, (
        f"sigma_center off: progress={progress}, got {c['sigma_center']}, want {expected_center}"
    )


@pytest.mark.parametrize("progress", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_sigma_sample_within_gate(sample_curriculum, progress):
    """±gate window around center, clipped to [0, 1]."""
    for _ in range(20):
        c = sample_curriculum(progress, **V2_DEFAULTS)
        gate = V2_DEFAULTS["sigma_gate"]
        lo = max(0.0, c["sigma_center"] - gate)
        hi = min(1.0, c["sigma_center"] + gate)
        assert lo <= c["sigma_sample"] <= hi, (
            f"sigma_sample {c['sigma_sample']} outside [{lo}, {hi}] at progress={progress}"
        )


@pytest.mark.parametrize("progress", [0.0, 0.5, 1.0])
def test_rollout_center_is_linear(sample_curriculum, progress):
    c = sample_curriculum(progress, **V2_DEFAULTS)
    expected = 0.20 + 0.40 * progress
    assert abs(c["rollout_center"] - expected) < 1e-9


@pytest.mark.parametrize("progress", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_rollout_sample_within_gate(sample_curriculum, progress):
    for _ in range(20):
        c = sample_curriculum(progress, **V2_DEFAULTS)
        gate = V2_DEFAULTS["rollout_gate"]
        lo = max(0.0, c["rollout_center"] - gate)
        hi = min(1.0, c["rollout_center"] + gate)
        assert lo <= c["rollout_sample"] <= hi


@pytest.mark.parametrize("progress", [0.0, 0.5, 1.0])
def test_n_sample_is_integer(sample_curriculum, progress):
    c = sample_curriculum(progress, **V2_DEFAULTS)
    assert isinstance(c["n_sample"], int)
    assert 1 <= c["n_sample"] <= 7, f"n_sample {c['n_sample']} outside [1, 7]"


def test_n_sample_within_gate():
    """At progress=0, center=2, ±1 gate -> samples in {1, 2, 3} (clipped).
    At progress=1, center=5, ±1 gate -> samples in {4, 5, 6}."""
    s = _load_sampler()
    samples_start = [s(0.0, **V2_DEFAULTS)["n_sample"] for _ in range(50)]
    samples_end = [s(1.0, **V2_DEFAULTS)["n_sample"] for _ in range(50)]
    assert all(1 <= n <= 3 for n in samples_start), f"start samples: {sorted(set(samples_start))}"
    assert all(4 <= n <= 6 for n in samples_end), f"end samples: {sorted(set(samples_end))}"
    # Sanity: there should be variance.
    assert len(set(samples_start)) > 1 or len(set(samples_end)) > 1, (
        "n_sample never varied across 50 draws -- gate isn't working"
    )


def test_gate_zero_disables_sampling():
    """When all gates are 0, sample == center exactly."""
    s = _load_sampler()
    deterministic = dict(
        noise_range_low=0.50, noise_range_high=0.90, sigma_gate=0.0,
        rollout_low=0.20, rollout_high=0.60, rollout_gate=0.0,
        n_min=2, n_max=5, n_gate=0,
    )
    for progress in [0.0, 0.5, 1.0]:
        c = s(progress, **deterministic)
        assert c["sigma_sample"] == c["sigma_center"]
        assert c["rollout_sample"] == c["rollout_center"]
        assert c["n_sample"] == round(c["n_center"])


def test_progress_clipped_to_unit_interval():
    """Out-of-range progress is clipped to [0, 1]."""
    s = _load_sampler()
    c_minus = s(-0.5, **V2_DEFAULTS)
    c_plus = s(1.5, **V2_DEFAULTS)
    c_zero = s(0.0, **V2_DEFAULTS)
    c_one = s(1.0, **V2_DEFAULTS)
    assert abs(c_minus["sigma_center"] - c_zero["sigma_center"]) < 1e-9
    assert abs(c_plus["sigma_center"] - c_one["sigma_center"]) < 1e-9
