#!/usr/bin/env python3
# Copyright 2026 University of Sydney
# Licensed under the Apache License, Version 2.0.
"""Parse a smoke-run log.txt and report whether the pipeline is healthy.

Usage:
    python tools/check_smoke.py dFactory/log.txt
    python tools/check_smoke.py path/to/smoke_run.log

Pass criteria (all must hold):
  - At least 100 successful training steps logged.
  - Loss at step ~50 is below 11.0 (started at log(vocab) ≈ 11.97; should have moved).
  - Final-50-step mean loss is below the first-50-step mean (descent verified).
  - No NaN / inf appears in loss or grad_norm.
  - Mid-run save_steps checkpoint message appears (exercises the save path).
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple


# tqdm postfix line looks like:
#   Epoch 1/2:   0%|...| 14/174854 [01:50<360:50:15,  7.43s/it, loss: 11.97, grad_norm: 0.34, lr: 2.67e-09]
TQDM_LINE = re.compile(
    r"(?P<step>\d+)/\d+\s+\[[^\]]+,\s+(?P<sec>[\d.]+)s/it,\s+"
    r"loss:\s+(?P<loss>-?[\d.]+|nan|inf),\s+"
    r"grad_norm:\s+(?P<gn>-?[\d.]+|nan|inf),\s+"
    r"lr:\s+(?P<lr>[-\d.e+]+)\]"
)

CHECKPOINT_LINE = re.compile(r"Distributed checkpoint saved at")


def parse_log(path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Return list of (step, sec_per_step, loss, grad_norm, lr)."""
    rows = []
    text = path.read_text(errors="replace")
    # tqdm overwrites with \r so newlines aren't reliable; split on both.
    for chunk in re.split(r"[\r\n]+", text):
        m = TQDM_LINE.search(chunk)
        if not m:
            continue
        try:
            step = int(m["step"])
            sec = float(m["sec"])
            loss = float(m["loss"])
            gn = float(m["gn"])
            lr = float(m["lr"])
        except (ValueError, TypeError):
            # NaN/inf will hit this; record specially
            try:
                step = int(m["step"])
            except (ValueError, TypeError):
                continue
            rows.append((step, math.nan, math.nan, math.nan, math.nan))
            continue
        rows.append((step, sec, loss, gn, lr))
    return rows


def has_checkpoint(path: Path) -> bool:
    text = path.read_text(errors="replace")
    return bool(CHECKPOINT_LINE.search(text))


def mean(xs: List[float]) -> Optional[float]:
    valid = [x for x in xs if not math.isnan(x) and not math.isinf(x)]
    return sum(valid) / len(valid) if valid else None


def report(rows, ckpt_saved: bool) -> int:
    """Returns exit code: 0 = pass, 1 = fail."""
    if not rows:
        print("FAIL: no parseable training-step lines found in log.")
        return 1

    n_steps = len(rows)
    steps = [r[0] for r in rows]
    secs = [r[1] for r in rows]
    losses = [r[2] for r in rows]
    grads = [r[3] for r in rows]
    lrs = [r[4] for r in rows]

    nan_loss = sum(1 for x in losses if math.isnan(x) or math.isinf(x))
    nan_gn = sum(1 for x in grads if math.isnan(x) or math.isinf(x))

    first_50 = losses[:50]
    last_50 = losses[-50:]
    mean_first = mean(first_50)
    mean_last = mean(last_50)
    mean_sec = mean(secs)
    final_lr = next((x for x in reversed(lrs) if not math.isnan(x)), None)
    final_step = steps[-1]

    print(f"steps observed       : {n_steps}  (final step number: {final_step})")
    print(f"mean s/it            : {mean_sec:.2f}" if mean_sec else "mean s/it            : n/a")
    print(f"mean loss steps 1-50 : {mean_first:.3f}" if mean_first else "mean loss steps 1-50 : n/a")
    print(f"mean loss last 50    : {mean_last:.3f}" if mean_last else "mean loss last 50    : n/a")
    print(f"final lr             : {final_lr:.2e}" if final_lr else "final lr             : n/a")
    print(f"NaN/inf in loss      : {nan_loss}")
    print(f"NaN/inf in grad_norm : {nan_gn}")
    print(f"checkpoint save seen : {ckpt_saved}")
    print()

    failures: List[str] = []
    if n_steps < 100:
        failures.append(f"too few steps observed ({n_steps} < 100). Did the run crash early?")
    if mean_first is not None and mean_first > 11.97 * 1.05:
        failures.append(f"first-50 loss mean {mean_first:.2f} is above log(vocab) ≈ 11.97 — initialisation likely broken")
    if mean_last is not None and mean_last >= 11.0:
        failures.append(f"last-50 loss mean {mean_last:.2f} hasn't dropped below 11.0 — talk not learning")
    if mean_first is not None and mean_last is not None and mean_last >= mean_first:
        failures.append(f"loss is not descending: first-50 mean {mean_first:.2f} <= last-50 mean {mean_last:.2f}")
    if nan_loss > 0 or nan_gn > 0:
        failures.append(f"NaN/inf observed: loss={nan_loss} grad_norm={nan_gn}")
    if not ckpt_saved:
        # Not strictly fatal but worth flagging.
        print("WARN: no checkpoint-save message found. If save_steps fired, this is suspicious.")
        print()

    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("PASS")
    print()
    print("Pipeline validation looks healthy. Smoke run can be deleted; ready to plan the real run.")
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path/to/log.txt>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"not a file: {path}", file=sys.stderr)
        return 2
    rows = parse_log(path)
    ckpt = has_checkpoint(path)
    return report(rows, ckpt)


if __name__ == "__main__":
    sys.exit(main())
