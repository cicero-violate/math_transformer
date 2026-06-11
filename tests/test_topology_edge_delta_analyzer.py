from __future__ import annotations

import csv

from src.topology_edge_delta_analyzer import (
    analyze_edge_delta_records,
    format_compact_table,
    format_polarity_table,
    write_edge_delta_summary,
)


def _record(sample_id, outcome, target, hand_pred, learned_pred, removed, extra):
    return {
        "sample_id": sample_id,
        "outcome": outcome,
        "target_expert": target,
        "dense_pred": "affine_expert",
        "hand_pred": hand_pred,
        "learned_pred": learned_pred,
        "removed_edges": removed,
        "extra_edges": extra,
    }


def test_analyze_edge_delta_records_counts_patterns_and_scores():
    records = [
        _record(60, "learned_win", "affine_expert", "matmul_expert", "affine_expert", [{"src_label": "add(x, y)", "dst_label": "matmul(A, x)", "score": 2.0}], [{"src_label": "add(x, y)", "dst_label": "x", "score": 0.5}]),
        _record(61, "learned_loss", "generic_expert", "generic_expert", "affine_expert", [{"src_label": "mul(x, y)", "dst_label": "y", "score": 1.0}], [{"src_label": "mul(x, y)", "dst_label": "add(x, y)", "score": -0.25}, {"src_label": "mul(x, y)", "dst_label": "z", "score": -0.5}]),
    ]
    summary = analyze_edge_delta_records(records)
    assert summary["n_records"] == 2
    assert summary["outcome_counts"] == {"learned_win": 1, "learned_loss": 1}
    assert summary["target_expert_counts"]["affine_expert"] == 1
    assert summary["prediction_flip_counts"]["matmul_expert->affine_expert"] == 1
    assert summary["edge_count_summary"]["all"]["extra_count"]["median"] == 1.5
    assert summary["edge_score_summary"]["all"]["extra_edges"]["min"] == -0.5
    assert summary["recurring_edge_kind_patterns"]["all"]["extra_edges:mul->leaf"] == 1
    assert summary["top_win_edge_kind_polarity"][0]["polarity"] > 0.0
    assert summary["top_loss_edge_kind_polarity"][0]["polarity"] < 0.0
    assert summary["compact_records"][0]["removed_count"] == 1


def test_edge_kind_polarity_normalizes_by_win_and_loss_occurrences():
    records = [
        _record(1, "learned_win", "generic_expert", "affine_expert", "generic_expert", [], [{"src_label": "add(x,y)", "dst_label": "x"}]),
        _record(2, "learned_win", "generic_expert", "affine_expert", "generic_expert", [], [{"src_label": "add(a,b)", "dst_label": "a"}]),
        _record(3, "learned_loss", "affine_expert", "generic_expert", "affine_expert", [], [{"src_label": "mul(x,y)", "dst_label": "x"}]),
    ]
    summary = analyze_edge_delta_records(records)
    rows = {row["pattern"]: row for row in summary["edge_kind_polarity"]}
    assert rows["extra_edges:add->leaf"]["win_count"] == 2
    assert rows["extra_edges:add->leaf"]["loss_count"] == 0
    assert rows["extra_edges:add->leaf"]["polarity"] == 1.0
    assert rows["extra_edges:mul->leaf"]["polarity"] == -1.0
    table = format_polarity_table(summary)
    assert "record_polarity" in table
    assert "extra_edges:add->leaf" in table


def test_format_compact_table_and_csv_write(tmp_path):
    records = [_record(60, "learned_win", "affine_expert", "matmul_expert", "affine_expert", [], [{"score": 1.0}])]
    table = format_compact_table(records)
    assert "sample_id" in table
    assert "learned_win" in table
    assert "removed_count" in table
    summary = analyze_edge_delta_records(records)
    json_out = tmp_path / "summary.json"
    csv_out = tmp_path / "summary.csv"
    write_edge_delta_summary(summary, json_out=json_out, csv_out=csv_out)
    assert json_out.exists()
    with csv_out.open() as f:
        rows = list(csv.DictReader(f))
    assert any(row["section"] == "outcome_counts" for row in rows)
    assert any(row["section"] == "edge_kind_polarity" for row in rows)
