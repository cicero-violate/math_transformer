"""Tests for topology_depth_sweep.py — P0.4 depth sweep at fixed K."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.topology_depth_sweep import (
    DEFAULT_K,
    DEFAULT_L_VALUES,
    SCHEMA_VERSION,
    DepthSweepRow,
    format_depth_sweep_table,
    write_depth_sweep_summary,
)


# ── DepthSweepRow ─────────────────────────────────────────────────────────────

def test_depth_sweep_row_as_dict():
    row = DepthSweepRow(
        l=2, k=4,
        topology_mode="middle_preserving_topk",
        route_accuracy=0.95,
        generic_accuracy=0.91,
        affine_accuracy=0.88,
        n_examples=1786,
    )
    d = row.as_dict()
    assert d["l"] == 2
    assert d["k"] == 4
    assert d["route_accuracy"] == pytest.approx(0.95)
    assert d["generic_accuracy"] == pytest.approx(0.91)
    assert d["affine_accuracy"] == pytest.approx(0.88)
    assert "notes" in d


def test_depth_sweep_row_none_accuracies():
    row = DepthSweepRow(
        l=1, k=4,
        topology_mode="scored_topk",
        route_accuracy=0.90,
        generic_accuracy=None,
        affine_accuracy=None,
        n_examples=100,
    )
    d = row.as_dict()
    assert d["generic_accuracy"] is None
    assert d["affine_accuracy"] is None


# ── format_depth_sweep_table ──────────────────────────────────────────────────

def _make_rows():
    return [
        DepthSweepRow(l=l, k=4, topology_mode="middle_preserving_topk",
                      route_accuracy=0.80 + l * 0.01,
                      generic_accuracy=0.75 + l * 0.01,
                      affine_accuracy=0.70 + l * 0.01,
                      n_examples=1786)
        for l in [1, 2, 4, 8]
    ]


def test_format_table_has_header():
    rows = _make_rows()
    table = format_depth_sweep_table(rows)
    assert "L" in table
    assert "K" in table
    assert "route_acc" in table


def test_format_table_has_rows():
    rows = _make_rows()
    table = format_depth_sweep_table(rows)
    lines = table.strip().split("\n")
    # header + sep + 4 data rows = 6
    assert len(lines) >= 6


def test_format_table_shows_na_for_none():
    rows = [DepthSweepRow(l=1, k=4, topology_mode="m", route_accuracy=0.9,
                          generic_accuracy=None, affine_accuracy=None, n_examples=10)]
    table = format_depth_sweep_table(rows)
    assert "N/A" in table


def test_format_table_each_l_present():
    rows = _make_rows()
    table = format_depth_sweep_table(rows)
    for l in [1, 2, 4, 8]:
        assert str(l) in table


# ── write_depth_sweep_summary ─────────────────────────────────────────────────

def test_write_summary_json(tmp_path):
    rows = _make_rows()
    json_path = tmp_path / "sweep.json"
    write_depth_sweep_summary(rows, json_out=json_path)
    data = json.loads(json_path.read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    assert len(data["rows"]) == 4
    assert data["rows"][0]["l"] == 1


def test_write_summary_csv(tmp_path):
    rows = _make_rows()
    csv_path = tmp_path / "sweep.csv"
    write_depth_sweep_summary(rows, csv_out=csv_path)
    with csv_path.open() as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 4
    assert "l" in reader[0]
    assert "route_accuracy" in reader[0]


def test_write_summary_both(tmp_path):
    rows = _make_rows()
    json_path = tmp_path / "sweep.json"
    csv_path = tmp_path / "sweep.csv"
    write_depth_sweep_summary(rows, json_out=json_path, csv_out=csv_path)
    assert json_path.exists()
    assert csv_path.exists()


# ── P0.3: k=None (dense/full mode) support ───────────────────────────────────

def test_depth_sweep_row_k_none():
    row = DepthSweepRow(
        l=2, k=None,
        topology_mode="full",
        route_accuracy=0.98,
        generic_accuracy=0.97,
        affine_accuracy=0.95,
        n_examples=1786,
    )
    d = row.as_dict()
    assert d["k"] is None


def test_format_table_k_none_no_crash():
    rows = [DepthSweepRow(l=2, k=None, topology_mode="full",
                          route_accuracy=0.98, generic_accuracy=0.97,
                          affine_accuracy=0.95, n_examples=1786)]
    table = format_depth_sweep_table(rows)
    assert "full" in table


def test_format_table_k_none_and_int_mixed():
    rows = [
        DepthSweepRow(l=2, k=None, topology_mode="full",
                      route_accuracy=0.98, generic_accuracy=None, affine_accuracy=None,
                      n_examples=100),
        DepthSweepRow(l=2, k=4, topology_mode="middle_preserving_topk",
                      route_accuracy=0.95, generic_accuracy=None, affine_accuracy=None,
                      n_examples=100),
    ]
    table = format_depth_sweep_table(rows)
    assert "full" in table
    assert "4" in table


# ── P0.3: correct expert key names ───────────────────────────────────────────

def test_extract_expert_acc_correct_key():
    """_extract_expert_acc must find generic_expert / affine_expert, not generic / affine."""
    from src.topology_depth_sweep import _extract_expert_acc

    mock_report = type("R", (), {
        "by_expert": {
            "generic_expert": {"correct": 80, "total": 100},
            "affine_expert": {"correct": 70, "total": 100},
        }
    })()

    g_acc, g_c, g_t = _extract_expert_acc(mock_report, "generic_expert")
    assert g_acc == pytest.approx(0.8)
    assert g_c == 80
    assert g_t == 100

    a_acc, a_c, a_t = _extract_expert_acc(mock_report, "affine_expert")
    assert a_acc == pytest.approx(0.7)

    # Old (wrong) keys must return None
    none_acc, _, _ = _extract_expert_acc(mock_report, "generic")
    assert none_acc is None
    none_acc2, _, _ = _extract_expert_acc(mock_report, "affine")
    assert none_acc2 is None


def test_run_topology_depth_sweep_uses_generic_expert_key(tmp_path):
    """run_topology_depth_sweep must look up generic_expert / affine_expert."""
    from src.topology_depth_sweep import run_topology_depth_sweep
    from unittest.mock import MagicMock, patch

    mock_report = MagicMock()
    mock_report.route_accuracy = 0.95
    mock_report.n_examples = 100
    mock_report.k = 4
    mock_report.by_expert = {
        "generic_expert": {"correct": 80, "total": 100},
        "affine_expert": {"correct": 70, "total": 100},
    }

    with patch("src.eval.run_quality_eval", return_value=[mock_report]):
        rows = run_topology_depth_sweep(
            examples_path="fake", checkpoint="fake", l_values=[1], fixed_k=4,
        )

    assert len(rows) == 1
    assert rows[0].generic_accuracy == pytest.approx(0.8)
    assert rows[0].affine_accuracy == pytest.approx(0.7)


# ── P0.3: L-sweep protocol note ──────────────────────────────────────────────

def test_run_topology_depth_sweep_adds_single_checkpoint_note():
    """Multi-L sweep adds single_checkpoint_l_sweep note to every row."""
    from src.topology_depth_sweep import run_topology_depth_sweep
    from unittest.mock import MagicMock, patch

    mock_report = MagicMock()
    mock_report.route_accuracy = 0.90
    mock_report.n_examples = 50
    mock_report.k = 4
    mock_report.by_expert = {}

    with patch("src.eval.run_quality_eval", return_value=[mock_report]):
        rows = run_topology_depth_sweep(
            examples_path="x", checkpoint="y", l_values=[1, 2],
        )

    assert all("single_checkpoint_l_sweep" in r.notes for r in rows)


def test_run_topology_depth_sweep_single_l_no_note():
    """Single-L sweep does not add the protocol-limitation note."""
    from src.topology_depth_sweep import run_topology_depth_sweep
    from unittest.mock import MagicMock, patch

    mock_report = MagicMock()
    mock_report.route_accuracy = 0.90
    mock_report.n_examples = 50
    mock_report.k = 4
    mock_report.by_expert = {}

    with patch("src.eval.run_quality_eval", return_value=[mock_report]):
        rows = run_topology_depth_sweep(
            examples_path="x", checkpoint="y", l_values=[2],
        )

    assert all(r.notes == "" for r in rows)


def test_write_summary_no_output_no_crash():
    rows = _make_rows()
    write_depth_sweep_summary(rows)  # neither json_out nor csv_out


# ── Constants ─────────────────────────────────────────────────────────────────

def test_default_l_values():
    assert DEFAULT_L_VALUES == [1, 2, 4, 8]


def test_default_k():
    assert DEFAULT_K == 4


def test_schema_version():
    assert SCHEMA_VERSION == "topology_depth_sweep.v1"


# ── run_topology_depth_sweep (stubbed eval) ───────────────────────────────────

def test_run_topology_depth_sweep_calls_eval_per_l(tmp_path):
    """Verify run_topology_depth_sweep calls run_quality_eval once per L value."""
    from src.topology_depth_sweep import run_topology_depth_sweep

    mock_report = MagicMock()
    mock_report.route_accuracy = 0.95
    mock_report.n_examples = 100
    mock_report.k = 4
    mock_report.by_expert = {
        "generic_expert": {"correct": 80, "total": 100},
        "affine_expert": {"correct": 70, "total": 100},
    }

    with patch("src.eval.run_quality_eval", return_value=[mock_report]) as mock_eval:
        rows = run_topology_depth_sweep(
            examples_path="fake/path",
            checkpoint="fake/ckpt",
            l_values=[1, 2],
            fixed_k=4,
        )

    assert mock_eval.call_count == 2  # once per L
    assert len(rows) == 2
    assert rows[0].l == 1
    assert rows[1].l == 2
    assert rows[0].generic_accuracy == pytest.approx(0.8)
    assert rows[0].affine_accuracy == pytest.approx(0.7)


def test_run_topology_depth_sweep_writes_outputs(tmp_path):
    from src.topology_depth_sweep import run_topology_depth_sweep

    mock_report = MagicMock()
    mock_report.route_accuracy = 0.90
    mock_report.n_examples = 50
    mock_report.k = 4
    mock_report.by_expert = {}

    json_path = tmp_path / "out.json"
    csv_path = tmp_path / "out.csv"

    with patch("src.eval.run_quality_eval", return_value=[mock_report]):
        run_topology_depth_sweep(
            examples_path="x",
            checkpoint="y",
            l_values=[1],
            json_out=json_path,
            csv_out=csv_path,
        )

    assert json_path.exists()
    assert csv_path.exists()
