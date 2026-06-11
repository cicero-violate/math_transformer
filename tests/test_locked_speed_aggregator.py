from __future__ import annotations

import csv

from src.locked_speed_aggregator import aggregate_locked_speed, format_speed_table, write_locked_speed_summary


def _artifact(seed: int, learned_k: int, hand_ms: float, learned_ms: float, dense_ms: float):
    return {
        "config": {"hand_k": 4, "learned_k": learned_k, "bench_seed": seed, "bench_n": 1024, "bench_node_mode": "trees"},
        "speed": {"hand_block_ms": hand_ms, "learned_block_ms": learned_ms},
        "reports": {"hand": {"full_block_ms": dense_ms}, "learned": {"full_block_ms": dense_ms}},
        "acceptance": {"quality_ok": True, "speed_ok": learned_ms <= hand_ms + 0.05, "strict_speed_ok": learned_ms < hand_ms},
    }


def test_aggregate_locked_speed_required_policies_and_gate():
    artifacts = [
        _artifact(0, 4, 10.0, 9.98, 12.0),
        _artifact(0, 8, 10.0, 9.90, 12.0),
        _artifact(1, 4, 11.0, 10.96, 13.0),
        _artifact(1, 8, 11.0, 10.90, 13.0),
    ]
    summary = aggregate_locked_speed(artifacts, tolerance_ms=0.05, min_pass_rate=1.0)
    assert summary["gate"]["speed_distribution_ok"] is True
    assert summary["policies"]["dense_full"]["median"] == 12.5
    assert summary["policies"]["hand_k4"]["p25"] == 10.25
    assert summary["policies"]["learned_k4"]["pass_rate"] == 1.0
    assert summary["policies"]["current_champion_k8"]["count"] == 2
    assert summary["pair_acceptance"]["quality_ok"]["mean"] == 1.0


def test_format_and_write_locked_speed_summary(tmp_path):
    summary = aggregate_locked_speed([_artifact(0, 4, 10.0, 9.99, 12.0)], required_policies=["dense_full", "hand_k4", "learned_k4"])
    table = format_speed_table(summary)
    assert "hand_k4" in table
    assert "pass_rate" in table
    json_out = tmp_path / "speed.json"
    csv_out = tmp_path / "speed.csv"
    write_locked_speed_summary(summary, json_out=json_out, csv_out=csv_out)
    assert json_out.exists()
    with csv_out.open() as f:
        rows = list(csv.DictReader(f))
    assert any(row["policy"] == "learned_k4" for row in rows)
