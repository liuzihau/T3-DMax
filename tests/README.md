# T3-D tests

## `test_anchor_leak.py` — mandatory pre-training check (brief sec 8.4)

Three checks:

| Test | Needs | What it verifies |
|---|---|---|
| `test_noisy_cannot_attend_to_clean` (parametrized) | nothing | The `block_diffusion_mask` returns False for every (q in [0, L), k in [L, 2L)) cell. Pure mask logic, no model. Always runs. |
| `test_noisy_self_attention_is_block_diagonal` | nothing | Within the noisy half, attention is block-diagonal at the configured `block_size`. |
| `test_anchor_invariant_to_clean_perturbation` (marked `slow`) | T3-D model importable | End-to-end: build a tiny ThinkTalkLLaDA2 model, run two forwards with the same noisy half but different clean halves, assert the noisy-half anchor is bit-identical (within fp32 tol). |
| `test_talk_logits_invariant_to_clean_perturbation` (marked `slow`) | T3-D model importable | Same as above but for talk logits. |

## Running

```bash
# From the T3-DMax repo root:
cd <T3-DMax>
PYTHONPATH=dFactory:dFactory/VeOmni:$PYTHONPATH pytest tests/test_anchor_leak.py -v

# To include the model-level golden tests:
PYTHONPATH=... pytest tests/test_anchor_leak.py -v -m slow

# To run everything:
PYTHONPATH=... pytest tests/test_anchor_leak.py -v --runslow
```

If any test fails, **do not start training** — fix the attention mask or the model before
proceeding. The cost of catching this here is hours; the cost of catching it after a full
GSM8K eval looks suspiciously good is weeks.

## What to do if `test_anchor_invariant_*` skips

The slow tests skip if `from models.think_talk_llada2 import ...` fails. This happens
if you're running them outside the cloned T3-DMax repo or without PYTHONPATH set. The
pure-mask test (`test_noisy_cannot_attend_to_clean`) still runs and gives partial
coverage — but the slow tests are the canonical check.
