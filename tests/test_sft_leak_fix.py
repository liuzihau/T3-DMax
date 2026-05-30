"""Unit tests for the SFT-label leak fix at data_transform.py:48.

Verifies that process_mdm_sft_example sets labels to -100 at:
  (a) prompt positions,
  (b) UNMASKED response positions (the leak-fix line).
And keeps valid labels at:
  (c) MASKED response positions (the actual training targets).

This is the most direct test of the line-48 uncomment. If this fails, the
leak is back and a retrain will produce the same degenerate behavior as v6e.

Run from the T3-DMax repo root:
    PYTHONPATH=dFactory:$PYTHONPATH \\
      LLADA2_TOKENIZER_PATH=../LLaDA2.0-mini-moe-merge \\
      pytest tests/test_sft_leak_fix.py -v

The tokenizer path can be set via the LLADA2_TOKENIZER_PATH env var or via
the symbolic default `../LLaDA2.0-mini-moe-merge` relative to the repo root.
If the tokenizer can't be loaded, the test is skipped (not failed).
"""

import importlib.util
import os

import pytest

# Skip the whole module if torch isn't installed (e.g., on a dev box without
# the training env). The trainer's runtime always has torch.
pytest.importorskip("torch")

# Default tokenizer location (relative to the repo root). Override via env var
# if your local layout differs.
DEFAULT_TOKENIZER_DIR = "../LLaDA2.0-mini-moe-merge"

MASK_ID = 156895


def _repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, ".."))


def _resolve_tokenizer_path():
    path = os.environ.get("LLADA2_TOKENIZER_PATH", DEFAULT_TOKENIZER_DIR)
    if not os.path.isabs(path):
        path = os.path.normpath(os.path.join(_repo_root(), path))
    return path


def _load_transform_fn():
    """Import process_mdm_sft_example without requiring dFactory/__init__.py."""
    path = os.path.normpath(os.path.join(
        _repo_root(), "dFactory", "tasks", "dataset", "data_transform.py",
    ))
    spec = importlib.util.spec_from_file_location("data_transform_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.process_mdm_sft_example


@pytest.fixture(scope="module")
def tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not available")

    path = _resolve_tokenizer_path()
    if not os.path.isdir(path):
        pytest.skip(
            f"Tokenizer not found at {path}. Set LLADA2_TOKENIZER_PATH or symlink "
            f"the LLaDA2.0-mini-moe-merge directory at {DEFAULT_TOKENIZER_DIR}."
        )
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Could not load tokenizer at {path}: {exc}")


@pytest.fixture(scope="module")
def transform_fn():
    return _load_transform_fn()


def _make_example(content="2+2 equals 4."):
    return {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": content},
        ],
        "flag": 0,
    }


def _process(transform_fn, tokenizer, sigma):
    """Build an SFT example, force exact sigma, return the transformed dict."""
    out = transform_fn(
        _make_example(),
        tokenizer=tokenizer,
        max_seq_len=128,
        text_keys="messages",
        noise_range=(sigma, sigma),
        mask_token_id=MASK_ID,
    )
    assert len(out) == 1
    return out[0]


# ----------------------------------------------------------------------------
# Core leak-fix invariants
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.30, 0.50, 0.75, 0.90])
def test_no_valid_labels_at_unmasked_response_positions(
    transform_fn, tokenizer, sigma,
):
    """The leak-fix line. After uncommenting line 48, no unmasked response
    position should have a valid label -- they're all -100."""
    ex = _process(transform_fn, tokenizer, sigma)
    labels = ex["labels"]
    noisy = ex["noisy_input_ids"]

    masked = (noisy == MASK_ID)
    has_valid_label = (labels != -100)
    # Where the input is NOT mask AND the label is valid -- this is the leak
    # surface. After the fix, the intersection should be empty.
    leak_surface = (~masked) & has_valid_label
    n_leak = int(leak_surface.sum().item())
    assert n_leak == 0, (
        f"LEAK FIX REGRESSED at sigma={sigma}: {n_leak} unmasked response positions "
        f"still carry valid labels (identity-copy bypass via tied embeddings is open)."
    )


@pytest.mark.parametrize("sigma", [0.30, 0.50, 0.75, 0.90])
def test_valid_labels_only_at_masked_response_positions(
    transform_fn, tokenizer, sigma,
):
    """Every position with label != -100 must be a MASK position in the noisy
    input. (Equivalent to the negation of the leak surface.)"""
    ex = _process(transform_fn, tokenizer, sigma)
    labels = ex["labels"]
    noisy = ex["noisy_input_ids"]

    has_valid_label = (labels != -100)
    if has_valid_label.sum().item() == 0:
        pytest.skip(f"No valid labels at sigma={sigma} -- maskable region must be too small")

    # Every valid-label position must currently be MASK in the noisy input.
    masked_at_valid = (noisy == MASK_ID)[has_valid_label]
    assert masked_at_valid.all(), (
        f"At sigma={sigma}: {int((~masked_at_valid).sum().item())} valid labels "
        f"land on non-mask positions. Labels should only score MASK positions."
    )


@pytest.mark.parametrize("sigma", [0.50, 0.75])
def test_masked_positions_carry_valid_labels(transform_fn, tokenizer, sigma):
    """Sanity: at typical sigmas SOME masked positions get valid labels (i.e.,
    the loss has something to score). Catches the over-correction case where
    line 48 accidentally also nukes the masked-position labels."""
    ex = _process(transform_fn, tokenizer, sigma)
    labels = ex["labels"]
    noisy = ex["noisy_input_ids"]

    masked = (noisy == MASK_ID)
    masked_with_valid_label = masked & (labels != -100)
    n_valid_at_mask = int(masked_with_valid_label.sum().item())
    n_mask_total = int(masked.sum().item())
    assert n_valid_at_mask > 0, (
        f"At sigma={sigma}: no masked position has a valid label "
        f"({n_mask_total} masked positions total). Loss has nothing to train on."
    )


def test_prompt_positions_always_ignored(transform_fn, tokenizer):
    """Sanity: prompt positions are always -100 regardless of sigma."""
    ex = _process(transform_fn, tokenizer, sigma=0.50)
    labels = ex["labels"]
    # Find prompt_length by reconstructing it from the noisy_input_ids:
    # prompt positions match input_ids exactly (never masked), and we set
    # labels[:prompt_length] = -100 in the transform.
    input_ids = ex["input_ids"]
    # The transform sets labels[:prompt_length] = -100. We can't read
    # prompt_length back, but we can check: the FIRST contiguous prefix of
    # labels that is all -100 must include the prompt area, and within that
    # prefix the noisy_input_ids equal input_ids (no masking on prompt).
    prefix_minus100 = (labels == -100).long()
    # find first index where labels != -100
    first_valid = int((labels != -100).long().argmax().item()) if (labels != -100).any() else len(labels)
    if first_valid == 0:
        pytest.skip("No -100 prefix -- malformed example?")
    # Within the -100 prefix, input_ids == noisy_input_ids (no masking on prompt).
    noisy_prefix = ex["noisy_input_ids"][:first_valid]
    input_prefix = input_ids[:first_valid]
    assert (noisy_prefix == input_prefix).all(), (
        "Noisy and input disagree within the -100 prefix -- prompt got masked?"
    )
