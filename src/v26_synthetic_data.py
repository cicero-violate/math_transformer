"""Deterministic synthetic task-family data for v26 decoder training."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def _require_split(split: str) -> None:
    if split not in {"train", "eval"}:
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")


def _record(sample_id: str, family: str, input_text: str, target: str, split: str) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "family": family,
        "input": input_text,
        "target": target,
        "split": split,
    }


def generate_arithmetic(
    n: int,
    *,
    split: str,
    seed: int = 0,
    a_range: tuple[int, int] = (0, 99),
    b_range: tuple[int, int] = (0, 99),
    ops: list[str] = ["+"],
) -> list[dict]:
    """
    Generate n arithmetic examples.
    Input:  "What is {a} + {b}?"
    Target: "reasoning: Add {a} and {b}.\nanswer: {a+b}"
    family: "arithmetic_short"
    Pairs must be unique within the generated set.
    ops may include "+", "-", "*" (no division, avoid negative targets for "-").
    """
    _require_split(split)
    allowed_ops = {"+", "-", "*"}
    bad_ops = [op for op in ops if op not in allowed_ops]
    if bad_ops:
        raise ValueError(f"unsupported arithmetic op: {bad_ops[0]!r}")
    candidates: list[tuple[int, int, str]] = []
    for op in ops:
        for a in range(a_range[0], a_range[1] + 1):
            for b in range(b_range[0], b_range[1] + 1):
                if op == "-" and a < b:
                    continue
                candidates.append((a, b, op))
    if n > len(candidates):
        raise ValueError(f"requested {n} examples but only {len(candidates)} unique arithmetic tuples are available")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    examples: list[dict] = []
    for i, (a, b, op) in enumerate(candidates[:n]):
        if op == "+":
            answer = a + b
            input_text = f"What is {a} + {b}?"
            target = f"reasoning: Add {a} and {b}.\nanswer: {answer}"
        elif op == "-":
            answer = a - b
            input_text = f"What is {a} - {b}?"
            target = f"reasoning: Subtract {b} from {a}.\nanswer: {answer}"
        else:
            answer = a * b
            input_text = f"What is {a} * {b}?"
            target = f"reasoning: Multiply {a} and {b}.\nanswer: {answer}"
        examples.append(_record(f"arith_{split}_{i:04d}", "arithmetic_short", input_text, target, split))
    return examples


def generate_algebra(
    n: int,
    *,
    split: str,
    seed: int = 0,
    coeff_range: tuple[int, int] = (1, 20),
    variables: list[str] = ["x", "y", "z"],
) -> list[dict]:
    """
    Generate n single-variable term-combination examples.
    Input:  "Simplify {c}x + {d}x. Return the simplified expression."
    Target: "reasoning: Combine like terms (both contain x).\nanswer: {c+d}x"
    family: "symbolic_short"
    Pairs (c, d, var) must be unique within the generated set.
    """
    _require_split(split)
    candidates = [
        (c, d, var)
        for var in variables
        for c in range(coeff_range[0], coeff_range[1] + 1)
        for d in range(coeff_range[0], coeff_range[1] + 1)
    ]
    if n > len(candidates):
        raise ValueError(f"requested {n} examples but only {len(candidates)} unique algebra tuples are available")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    examples: list[dict] = []
    for i, (c, d, var) in enumerate(candidates[:n]):
        input_text = f"Simplify {c}{var} + {d}{var}. Return the simplified expression."
        target = f"reasoning: Combine like terms (both contain {var}).\nanswer: {c + d}{var}"
        examples.append(_record(f"symbolic_{split}_{i:04d}", "symbolic_short", input_text, target, split))
    return examples


def _split_disjoint(
    candidates: list[Any],
    n_train: int,
    n_eval: int,
    seed: int,
) -> tuple[list[Any], list[Any]]:
    if n_train + n_eval > len(candidates):
        raise ValueError(f"requested {n_train + n_eval} examples but only {len(candidates)} unique tuples are available")
    rng = random.Random(seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return shuffled[:n_train], shuffled[n_train:n_train + n_eval]


def generate_task_family_dataset(
    family: str,
    n_train: int,
    n_eval: int,
    *,
    seed: int = 0,
) -> dict[str, list[dict]]:
    """
    Returns {"train": [...], "eval": [...]} for the given family.
    Train and eval sets must be disjoint.
    Supported families: "arithmetic_short", "symbolic_short".
    Raise ValueError for unknown family.
    """
    if family == "arithmetic_short":
        candidates = [(a, b, "+") for a in range(0, 100) for b in range(0, 100)]
        train_tuples, eval_tuples = _split_disjoint(candidates, n_train, n_eval, seed)
        train = [
            _record(
                f"arith_train_{i:04d}",
                family,
                f"What is {a} + {b}?",
                f"reasoning: Add {a} and {b}.\nanswer: {a + b}",
                "train",
            )
            for i, (a, b, _op) in enumerate(train_tuples)
        ]
        eval_rows = [
            _record(
                f"arith_eval_{i:04d}",
                family,
                f"What is {a} + {b}?",
                f"reasoning: Add {a} and {b}.\nanswer: {a + b}",
                "eval",
            )
            for i, (a, b, _op) in enumerate(eval_tuples)
        ]
        return {"train": train, "eval": eval_rows}
    if family == "symbolic_short":
        variables = ["x", "y", "z"]
        candidates = [(c, d, var) for var in variables for c in range(1, 21) for d in range(1, 21)]
        train_tuples, eval_tuples = _split_disjoint(candidates, n_train, n_eval, seed)
        train = [
            _record(
                f"symbolic_train_{i:04d}",
                family,
                f"Simplify {c}{var} + {d}{var}. Return the simplified expression.",
                f"reasoning: Combine like terms (both contain {var}).\nanswer: {c + d}{var}",
                "train",
            )
            for i, (c, d, var) in enumerate(train_tuples)
        ]
        eval_rows = [
            _record(
                f"symbolic_eval_{i:04d}",
                family,
                f"Simplify {c}{var} + {d}{var}. Return the simplified expression.",
                f"reasoning: Combine like terms (both contain {var}).\nanswer: {c + d}{var}",
                "eval",
            )
            for i, (c, d, var) in enumerate(eval_tuples)
        ]
        return {"train": train, "eval": eval_rows}
    raise ValueError(f"unknown task family: {family}")


def write_dataset(examples: list[dict], path: str | Path) -> None:
    """Write examples as newline-delimited JSON to path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, sort_keys=True) + "\n")


def load_dataset(path: str | Path) -> list[dict]:
    """Load newline-delimited JSON from path. Return list of dicts."""
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows
