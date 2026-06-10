from __future__ import annotations

import json
from pathlib import Path

from src.eval_topology_scorer import evaluate_topology_scorer
from src.synthetic_data import generate_hard_records, write_jsonl
from src.train_topology_scorer import train_topology_scorer


def test_evaluate_topology_scorer_smoke(tmp_path: Path):
    data = tmp_path / "hard.jsonl"
    ckpt = tmp_path / "scorer.pt"
    records = generate_hard_records(10, seed=41, split="train", route_fraction=1.0, max_depth=3)
    write_jsonl(data, records)
    train_topology_scorer(
        examples_path=str(data),
        save_checkpoint=str(ckpt),
        max_steps=3,
        max_examples=5,
        hidden_dim=8,
        target_k=4,
        eval_k=3,
        device="cpu",
        log_interval=2,
    )
    trace_path = tmp_path / "topology_traces.jsonl"
    summary = evaluate_topology_scorer(
        examples_path=str(data),
        checkpoint=str(ckpt),
        eval_k=3,
        target_k=4,
        device="cpu",
        max_examples=5,
        trace_output=str(trace_path),
    )

    assert summary["examples"] == 5
    assert summary["trace_output"] == str(trace_path)
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert len(rows) == 5
    assert rows[0]["feature_schema"] == "topology_edge_features.v1"
    assert rows[0]["features"]["shape"][-1] == 10
    assert "scores" in rows[0]
    assert "pred_topology" in rows[0]
    assert "overlap" in rows[0]
    assert 0.0 <= float(summary["mean_row_recall"]) <= 1.0
    assert 0.0 <= float(summary["mean_row_precision"]) <= 1.0
    assert summary["row_cap_violations"] == 0
