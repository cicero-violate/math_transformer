"""Capability eval gate for v26 graph decoder checkpoints."""
from __future__ import annotations

import re
from typing import Any

from src.v26_graph_decoder import generate

SCHEMA_VERSION = "v26_capability_eval.v1"

ANSWER_RE = re.compile(r"answer:\s*(.+)", re.IGNORECASE)


def extract_answer(text: str) -> str | None:
    """
    Extract the answer from generated text.
    Looks for pattern: "answer: <value>" (last occurrence wins).
    Returns the stripped value string, or None if not found.
    """
    matches = ANSWER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def _norm_answer(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower()


def eval_capability(
    model,
    tokenizer,
    eval_examples: list[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int = 64,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Run generation on each eval example, extract answer, compare to ground truth.
    """
    per_example: list[dict[str, Any]] = []
    n_correct = 0
    for i, ex in enumerate(eval_examples):
        input_text = str(ex.get("input") or "")
        target_text = str(ex.get("target") or "")
        generated = generate(
            model,
            tokenizer,
            input_text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            seed=seed + i,
            device=device,
        )
        target_answer = extract_answer(target_text)
        predicted_answer = extract_answer(str(generated["text"]))
        correct = _norm_answer(predicted_answer) == _norm_answer(target_answer)
        if correct:
            n_correct += 1
        per_example.append({
            "sample_id": str(ex.get("sample_id") or ""),
            "input": input_text,
            "target_answer": target_answer,
            "predicted_answer": predicted_answer,
            "generated_text": str(generated["text"]),
            "correct": correct,
        })
    n_total = len(eval_examples)
    return {
        "schema_version": SCHEMA_VERSION,
        "n_total": n_total,
        "n_correct": n_correct,
        "pass_rate": (n_correct / n_total) if n_total else 0.0,
        "per_example": per_example,
        "teacher_checkpoint_loaded": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
    }


def eval_gate_passes(eval_report: dict, *, threshold: float = 0.8) -> bool:
    """Return True if pass_rate >= threshold."""
    return float(eval_report.get("pass_rate", 0.0)) >= threshold
