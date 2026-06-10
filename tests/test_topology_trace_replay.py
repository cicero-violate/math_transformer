from __future__ import annotations

import json
from pathlib import Path

from src.topology_trace_replay import (
    is_failure_trace,
    replay_score,
    select_replay_candidates,
    summarize_replay_candidates,
    write_jsonl,
)


def _row(sample_id: int, *, learned_correct: bool, missing: int = 0, generic: bool = False):
    return {
        "sample_id": sample_id,
        "domain": "math",
        "expr": f"expr_{sample_id}",
        "nodes_hash": f"h{sample_id}",
        "scorer_checkpoint": "runs/checkpoints/scorer.pt",
        "feature_schema": "topology_edge_features.v1",
        "topology_config": {"learned_k": 8},
        "prediction": {
            "target_expert": "generic_expert" if generic else "affine_expert",
            "target_expert_id": 1 if generic else 0,
            "dense_correct": True,
            "hand_correct": True,
            "learned_correct": learned_correct,
            "learned_pred": "wrong" if not learned_correct else "affine_expert",
            "learned_pred_id": 2 if not learned_correct else 0,
        },
        "agreement": {
            "hidden_l1": 0.5 if not learned_correct else 0.01,
            "hidden_cos": 0.8 if not learned_correct else 0.99,
            "logit_kl": 0.2 if not learned_correct else 0.001,
        },
        "overlap": {
            "missing_edges": missing,
            "extra_edges": 1,
        },
        "diagnostics": {
            "failure": not learned_correct,
            "failure_type": "route_miss" if not learned_correct else None,
            "is_generic_expert": generic,
            "missing_edge_count": missing,
            "extra_edge_count": 1,
        },
    }


def test_trace_replay_selects_and_ranks_failures(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    rows = [
        _row(0, learned_correct=True),
        _row(1, learned_correct=False, missing=2, generic=False),
        _row(2, learned_correct=False, missing=5, generic=True),
    ]
    write_jsonl(trace, rows)

    selected = select_replay_candidates([trace], max_records=10, failures_only=True)
    assert [r["sample_id"] for r in selected] == [2, 1]
    assert selected[0]["failure"] is True
    assert selected[0]["is_generic_expert"] is True
    assert selected[0]["missing_edge_count"] == 5
    assert selected[0]["source_trace"] == str(trace)
    assert replay_score(rows[2]) > replay_score(rows[1])
    assert is_failure_trace(rows[0]) is False
    assert is_failure_trace(rows[1]) is True


def test_trace_replay_summary_and_output(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    out = tmp_path / "replay.jsonl"
    write_jsonl(trace, [
        _row(1, learned_correct=False, missing=2),
        _row(2, learned_correct=False, missing=4, generic=True),
    ])
    selected = select_replay_candidates([trace], max_records=1)
    assert len(selected) == 1
    count = write_jsonl(out, selected)
    assert count == 1
    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert written[0]["sample_id"] == 2

    summary = summarize_replay_candidates(selected)
    assert summary["records"] == 1
    assert summary["generic_expert_records"] == 1
    assert summary["by_expert"] == {"generic_expert": 1}
    assert summary["max_replay_score"] >= summary["mean_replay_score"] > 0
