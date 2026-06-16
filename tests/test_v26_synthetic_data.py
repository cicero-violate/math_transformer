from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.v26_synthetic_data import (
    generate_algebra,
    generate_arithmetic,
    generate_task_family_dataset,
    load_dataset,
    write_dataset,
)


ARITH_RE = re.compile(r"What is (\d+) ([+\-*]) (\d+)\?")
ANSWER_RE = re.compile(r"answer:\s*([0-9]+[a-z]?)")
ALG_RE = re.compile(r"Simplify (\d+)([xyz]) \+ (\d+)\2\.")


def test_generate_arithmetic_schema_count():
    rows = generate_arithmetic(10, split="train", seed=3)
    assert len(rows) == 10
    for row in rows:
        assert set(row) == {"sample_id", "family", "input", "target", "split"}
        assert row["family"] == "arithmetic_short"
        assert row["split"] == "train"


def test_generate_arithmetic_answers_correct_spot_check():
    rows = generate_arithmetic(20, split="eval", seed=4, ops=["+", "-", "*"])
    for row in rows:
        match = ARITH_RE.match(row["input"])
        assert match is not None
        a = int(match.group(1))
        op = match.group(2)
        b = int(match.group(3))
        expected = {"+": a + b, "-": a - b, "*": a * b}[op]
        assert f"answer: {expected}" in row["target"]


def test_train_eval_splits_disjoint_arithmetic():
    dataset = generate_task_family_dataset("arithmetic_short", 80, 40, seed=5)
    train_inputs = {row["input"] for row in dataset["train"]}
    eval_inputs = {row["input"] for row in dataset["eval"]}
    assert train_inputs.isdisjoint(eval_inputs)


def test_generator_seeded_deterministic():
    a = generate_arithmetic(25, split="train", seed=9)
    b = generate_arithmetic(25, split="train", seed=9)
    c = generate_arithmetic(25, split="train", seed=10)
    assert a == b
    assert a != c


def test_generate_algebra_answers_correct():
    rows = generate_algebra(20, split="train", seed=8)
    for row in rows:
        match = ALG_RE.match(row["input"])
        assert match is not None
        c = int(match.group(1))
        var = match.group(2)
        d = int(match.group(3))
        answer = ANSWER_RE.search(row["target"])
        assert answer is not None
        assert answer.group(1) == f"{c + d}{var}"


def test_generate_task_family_dataset_unknown_family():
    with pytest.raises(ValueError):
        generate_task_family_dataset("unknown_family", 1, 1)


def test_write_load_dataset_roundtrip(tmp_path: Path):
    rows = generate_algebra(5, split="eval", seed=11)
    path = tmp_path / "eval.jsonl"
    write_dataset(rows, path)
    assert load_dataset(path) == rows
