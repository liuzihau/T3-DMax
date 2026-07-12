# Copyright 2026 University of Sydney. Apache-2.0.
"""Shared task registry for the decode drivers: gsm8k / math500 / algebra (Minerva) / asdiv.

Prompts are VERBATIM from the lm-eval task yamls under evaluations/tasks/ (what the official
eval_llada_dmax_math.sh runs), so accuracies are comparable with the lm-eval numbers:
  gsm8k   : "Question: {question}\\nLet's think step by step\\nAnswer:"          (gsm8k main, test)
  math500 : "{problem}\\nLet's think step by step, and put your final answer within \\boxed{}."
            (HuggingFaceH4/MATH-500, test; template inherited from minerva_math_algebra.yaml)
  algebra : same template                                (EleutherAI/hendrycks_math 'algebra', test)
  asdiv   : "Question: {body} {question}\\nLet's think step by step\\nAnswer:"  (EleutherAI/asdiv, validation)

`load_task` returns rows of {"prompt": chat-template-ready text, "question": source text (stored in the
prediction jsonl for the graders/debugging)}. Graders (val_*.py) match predictions to ground truth by ORDER,
so drivers must iterate rows as returned. Each driver stamps `"task"` into its jsonl rows so
summarize_baseline picks the right grader automatically.
"""

import json

_MINERVA_TMPL = "{problem}\nLet's think step by step, and put your final answer within \\boxed{{}}."

TASKS = {
    "gsm8k":   {"grader": "val_gsm8k.py"},
    "math500": {"grader": "val_math.py"},
    "algebra": {"grader": "val_algebra.py"},
    "asdiv":   {"grader": "val_asdiv.py"},
}


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_task(task, limit=None, gt_jsonl_path=None):
    """-> list of {"prompt", "question"} in dataset order (the graders match by order)."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; choose from {sorted(TASKS)}")
    if gt_jsonl_path:
        raw = _read_jsonl(gt_jsonl_path)
    else:
        from datasets import load_dataset
        if task == "gsm8k":
            raw = list(load_dataset("gsm8k", "main", split="test"))
        elif task == "math500":
            raw = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
        elif task == "algebra":
            raw = list(load_dataset("EleutherAI/hendrycks_math", "algebra", split="test"))
        else:  # asdiv
            raw = list(load_dataset("EleutherAI/asdiv", split="validation"))
    rows = []
    for ex in raw:
        if task == "gsm8k":
            q = ex["question"]
            prompt = f"Question: {q}\nLet's think step by step\nAnswer:"
        elif task in ("math500", "algebra"):
            q = ex["problem"]
            prompt = _MINERVA_TMPL.format(problem=q)
        else:  # asdiv: "{{body if body is defined}} {{question}}"
            body = str(ex.get("body", "") or "")
            q = (body + " " + ex["question"]).strip()
            prompt = f"Question: {q}\nLet's think step by step\nAnswer:"
        rows.append({"prompt": prompt, "question": q})
    if limit is not None:
        rows = rows[:limit]
    return rows


def grader_for(task):
    return TASKS[task]["grader"]
