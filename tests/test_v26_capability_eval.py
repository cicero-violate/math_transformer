from __future__ import annotations

from pathlib import Path

import torch

from src.v26_capability_eval import (
    SCHEMA_VERSION,
    eval_capability,
    eval_gate_passes,
    extract_answer,
)
from src.v26_graph_decoder import FullGraphDecoderConfig, FullGraphDecoderLM, GraphTokenizer
from src.v26_synthetic_data import generate_arithmetic


def _make_model_and_tokenizer():
    examples = generate_arithmetic(8, split="train", seed=0)
    tokenizer = GraphTokenizer.build(examples)
    config = FullGraphDecoderConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=32,
        hidden_dim=16,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        graph_bias_weight=0.0,
        n_graph_nodes=0,
        adjacency_name="qwen_topk_k2",
    )
    torch.manual_seed(0)
    return FullGraphDecoderLM(config), tokenizer


def test_extract_answer_returns_value():
    assert extract_answer("reasoning: Add 3 and 5.\nanswer: 8") == "8"


def test_extract_answer_returns_none():
    assert extract_answer("no answer here") is None


def test_eval_capability_tiny_model_valid_schema(tmp_path: Path):
    model, tokenizer = _make_model_and_tokenizer()
    eval_examples = generate_arithmetic(4, split="eval", seed=3)
    report = eval_capability(model, tokenizer, eval_examples, max_tokens=8, seed=2)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["n_total"] == 4
    assert 0 <= report["n_correct"] <= 4
    assert 0.0 <= report["pass_rate"] <= 1.0
    assert len(report["per_example"]) == 4
    for row in report["per_example"]:
        assert set(row) == {
            "sample_id",
            "input",
            "target_answer",
            "predicted_answer",
            "generated_text",
            "correct",
        }


def test_eval_gate_passes_true():
    assert eval_gate_passes({"pass_rate": 0.9}, threshold=0.8) is True


def test_eval_gate_passes_false():
    assert eval_gate_passes({"pass_rate": 0.7}, threshold=0.8) is False


def test_eval_report_safety_flags():
    model, tokenizer = _make_model_and_tokenizer()
    report = eval_capability(model, tokenizer, generate_arithmetic(2, split="eval", seed=4), max_tokens=4)
    assert report["teacher_checkpoint_loaded"] is False
    assert report["raw_weight_payload_in_graph"] is False
    assert report["bounded_active_adjacency"] is True
