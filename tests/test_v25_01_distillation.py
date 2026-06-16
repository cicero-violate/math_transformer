from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.v25_01_distillation import (
    DEFAULT_TEACHER_ID,
    build_distill_examples,
    extract_answer,
    filter_teacher_responses,
    generate_v25_01_prompt_rows,
    iter_jsonl,
    run_sparse_student_distill,
    write_default_distill_config,
    write_json,
    write_jsonl,
    write_v25_01_prompts,
)


def _write_teacher_responses(base: Path) -> None:
    rows = []
    for prompt in iter_jsonl(base / "teacher_prompts.jsonl"):
        expected = prompt["expected_answer"]
        response = f"reasoning: compact check\nanswer: {expected}"
        rows.append(
            {
                "sample_id": prompt["sample_id"],
                "teacher_id": DEFAULT_TEACHER_ID,
                "prompt_hash": prompt["prompt_hash"],
                "response": response,
                "answer_extracted": extract_answer(response),
                "latency_ms": 1,
                "eval_options": {"temperature": 0.0, "num_predict": 128},
            }
        )
    write_jsonl(base / "teacher_responses.jsonl", rows)


def _write_adjacency(path: Path) -> None:
    write_json(
        path,
        {
            "schema_version": "qwen_selected_adjacency.v1",
            "adjacency_name": "qwen_topk_k1",
            "k": 1,
            "fixed_adjacency": True,
            "edges": [
                {
                    "edge_id": "e0",
                    "src_id": "n0",
                    "dst_id": "n1",
                    "relation": "test",
                    "weight": 1.0,
                    "score_name": "unit",
                }
            ],
        },
    )


def test_generate_v25_01_prompts_is_deterministic_and_schema_complete():
    first = generate_v25_01_prompt_rows(16)
    second = generate_v25_01_prompt_rows(16)
    assert first == second
    assert {row["family"] for row in first} == {
        "arithmetic_short",
        "symbolic_short",
        "logic_short",
        "project_specific_math_transformer",
    }
    assert all(row["source"] == "synthetic_v25_01" for row in first)
    assert all(row["sample_id"] and row["prompt_hash"] for row in first)


def test_filter_and_build_distill_examples(tmp_path):
    write_default_distill_config(tmp_path)
    write_v25_01_prompts(tmp_path, n=12, force=True)
    _write_teacher_responses(tmp_path)
    quality = filter_teacher_responses(tmp_path, force=True)
    assert len(quality) == 12
    assert all(row["kept_for_training"] for row in quality)
    report = build_distill_examples(tmp_path, force=True)
    assert report["status"] == "distill_examples_ready"
    assert report["example_count"] == 12
    rows = list(iter_jsonl(tmp_path / "distill_examples.jsonl"))
    assert rows[0]["loss_mask"] == "target_only"
    assert rows[0]["target_token_count"] > 0


def test_filter_marks_bad_teacher_answer_failure(tmp_path):
    write_v25_01_prompts(tmp_path, n=4, force=True)
    _write_teacher_responses(tmp_path)
    rows = list(iter_jsonl(tmp_path / "teacher_responses.jsonl"))
    rows[0]["response"] = "reasoning: wrong\nanswer: 9999"
    rows[0]["answer_extracted"] = "9999"
    write_jsonl(tmp_path / "teacher_responses.jsonl", rows)
    quality = filter_teacher_responses(tmp_path, force=True)
    assert quality[0]["kept_for_training"] is False
    failures = list(iter_jsonl(tmp_path / "failure_cases.jsonl"))
    assert failures[0]["drop_reason"] == "expected_answer_mismatch"


def test_sparse_student_distill_writes_plan_artifacts_and_preserves_adjacency(tmp_path):
    write_default_distill_config(tmp_path)
    write_v25_01_prompts(tmp_path, n=12, force=True)
    _write_teacher_responses(tmp_path)
    filter_teacher_responses(tmp_path, force=True)
    build_distill_examples(tmp_path, force=True)
    adjacency = tmp_path / "adjacency.json"
    _write_adjacency(adjacency)
    before = adjacency.read_bytes()
    report = run_sparse_student_distill(
        tmp_path,
        adjacency=adjacency,
        student_config=Path("configs/sparse_student_v25_01.json"),
        train_steps=4,
        lr=0.25,
    )
    assert report["status"] == "v25_01_sparse_student_distill_ok"
    assert report["fixed_adjacency"] is True
    assert report["allow_rewiring"] is False
    assert report["adjacency_unchanged"] is True
    assert report["loss_text_final"] <= report["loss_text_initial"]
    assert adjacency.read_bytes() == before
    for artifact in report["artifacts"]:
        assert (tmp_path / artifact).exists()
    kd = json.loads((tmp_path / "kd_delta_report.json").read_text(encoding="utf-8"))
    assert kd["logit_kd_used"] is False


def test_sparse_student_distill_rejects_rewiring(tmp_path):
    write_v25_01_prompts(tmp_path, n=4, force=True)
    _write_teacher_responses(tmp_path)
    filter_teacher_responses(tmp_path, force=True)
    build_distill_examples(tmp_path, force=True)
    with pytest.raises(ValueError, match="allow_rewiring=false"):
        run_sparse_student_distill(tmp_path, allow_rewiring=True)
