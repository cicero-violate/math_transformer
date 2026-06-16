import json
from pathlib import Path

from src.v26_sparse_student_generate import (
    generate_from_token_bias,
    load_student_checkpoint,
    retrieve_nearest_response,
)


def _write_checkpoint(path: Path) -> None:
    path.write_text(json.dumps({
        "vocab": ["answer", ":", "42", "."],
        "token_logits": {"answer": 2.0, ":": 1.0, "42": 3.0, ".": 0.5},
        "adjacency_checksum": "abc",
    }))


def _write_examples(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    rows = [
        {"sample_id": "a", "family": "arithmetic_short", "input_text": "What is 6 times 7?", "target_text": "reasoning: 6 * 7 = 42\nanswer: 42"},
        {"sample_id": "b", "family": "logic_short", "input_text": "Choose item_2 from the list", "target_text": "reasoning: direct lookup\nanswer: item_2"},
    ]
    with (base / "distill_examples.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_load_checkpoint_and_token_bias_generation(tmp_path):
    ck = tmp_path / "student_checkpoint.json"
    _write_checkpoint(ck)
    checkpoint = load_student_checkpoint(ck)
    result = generate_from_token_bias(checkpoint, prompt="hello", max_tokens=4, greedy=True)
    assert result["status"] == "generated_from_token_bias"
    assert result["mode"] == "token_bias"
    assert result["tokens"][0] == "42"
    assert "not a prompt-conditioned" in result["capability_note"]


def test_nearest_response_retrieval(tmp_path):
    _write_examples(tmp_path)
    result = retrieve_nearest_response(prompt="What is 6 times 7?", teacher_artifacts=tmp_path)
    assert result["status"] == "retrieved_nearest_teacher_response"
    assert result["nearest_sample_id"] == "a"
    assert "answer: 42" in result["text"]
    assert result["nearest_score"] > 0
