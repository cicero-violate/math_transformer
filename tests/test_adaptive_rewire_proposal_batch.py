from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.adaptive_rewire_proposal_batch import (
    PROPOSAL_BATCH_FILENAME,
    PROPOSAL_BATCH_REPORT_FILENAME,
    load_proposal_batch_report,
    main,
    run_and_write_proposal_batch_report,
    validate_proposal_batch_report,
)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _utility_row(
    edge_id: str,
    source_node: str,
    target_node: str,
    edge_type: str,
    *,
    utility_score: float,
    archive_priority: float,
    activation_frequency_norm: float,
    closure_critical_flag: bool = False,
) -> dict:
    return {
        "schema_version": "adaptive_rewire_edge_utility.row.v1",
        "student_id": "qwen_style_tiny",
        "experiment_id": "v26_p10_edge_utility",
        "cycle_index": 1,
        "edge_id": edge_id,
        "source_node": source_node,
        "target_node": target_node,
        "edge_type": edge_type,
        "active": True,
        "in_initial_adjacency": True,
        "activation_count": 1,
        "trace_row_count": 3,
        "activation_frequency": activation_frequency_norm,
        "activation_frequency_norm": activation_frequency_norm,
        "gradient_norm": None,
        "loss_contribution": None,
        "error_correlation": None,
        "compute_cost": 1.0,
        "compute_cost_norm": 0.5,
        "source_prior_score": 1.0,
        "source_prior_score_norm": 0.5,
        "closure_critical_flag": closure_critical_flag,
        "utility_score": utility_score,
        "archive_priority": archive_priority,
        "add_priority": 0.0,
        "reason_codes": ["fixture"],
        "source_provenance": {"fixture": True},
    }


def _build_edge_utility_fixture(tmp_path: Path) -> Path:
    out = tmp_path / "edge_utility"
    rows = [
        _utility_row("e1", "a", "b", "attn", utility_score=0.20, archive_priority=0.85, activation_frequency_norm=0.10),
        _utility_row("e2", "b", "c", "mlp", utility_score=0.45, archive_priority=0.40, activation_frequency_norm=0.15),
        _utility_row("e3", "c", "d", "attn", utility_score=0.90, archive_priority=0.05, activation_frequency_norm=1.00),
        _utility_row("e4", "d", "e", "mlp", utility_score=0.60, archive_priority=0.20, activation_frequency_norm=0.70, closure_critical_flag=True),
    ]
    utility_path = out / "edge_utility.jsonl"
    _write_jsonl(utility_path, rows)
    report = {
        "schema_version": "adaptive_rewire_edge_utility.v1",
        "status": "edge_utility_report_ok",
        "source_contract_manifest": str(tmp_path / "contract.json"),
        "output_dir": str(out),
        "edge_utility_path": str(utility_path),
        "edge_count": len(rows),
        "active_edge_count": len(rows),
        "trace_row_count": 12,
        "missing_gradient_stats": True,
        "missing_loss_contribution": True,
        "missing_error_correlation": True,
        "normalized": True,
        "topology_mutated": False,
        "bounded_active_adjacency": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "next_stage": "proposal_batch_generator",
    }
    _write_json(tmp_path / "contract.json", {"schema_version": "fixture.contract.v1"})
    report_path = out / "edge_utility_report.json"
    _write_json(report_path, report)
    return report_path


def test_proposal_batch_happy_path_writes_bounded_non_mutating_proposals(tmp_path):
    report_path = _build_edge_utility_fixture(tmp_path)
    out = tmp_path / "proposal_batch"

    summary = run_and_write_proposal_batch_report(
        edge_utility_report=report_path,
        output_dir=out,
        proposal_budget=3,
        archive_priority_threshold=0.50,
        downweight_priority_threshold=0.35,
        keep_utility_threshold=0.75,
    )

    assert summary["status"] == "proposal_batch_generation_ok"
    assert (out / PROPOSAL_BATCH_FILENAME).exists()
    assert (out / PROPOSAL_BATCH_REPORT_FILENAME).exists()
    proposals = _read_jsonl(out / PROPOSAL_BATCH_FILENAME)
    assert len(proposals) <= 3
    for row in proposals:
        assert row["auto_accepted"] is False
        assert row["accepted"] is False
        assert row["rejected"] is False
        assert row["topology_mutated"] is False
        assert row["reason_codes"]
        assert row["rollback_data"]
    report = load_proposal_batch_report(out)
    assert validate_proposal_batch_report(report)["proposal_count"] == len(proposals)


def test_proposal_batch_is_deterministic(tmp_path):
    report_path = _build_edge_utility_fixture(tmp_path)
    out_a = tmp_path / "proposal_a"
    out_b = tmp_path / "proposal_b"

    run_and_write_proposal_batch_report(edge_utility_report=report_path, output_dir=out_a, proposal_budget=4)
    run_and_write_proposal_batch_report(edge_utility_report=report_path, output_dir=out_b, proposal_budget=4)

    assert (out_a / PROPOSAL_BATCH_FILENAME).read_text(encoding="utf-8") == (
        out_b / PROPOSAL_BATCH_FILENAME
    ).read_text(encoding="utf-8")


def test_proposal_batch_rejects_unsafe_edge_utility_report(tmp_path):
    report_path = _build_edge_utility_fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["topology_mutated"] = True
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="topology_mutated must be false"):
        run_and_write_proposal_batch_report(edge_utility_report=report_path, output_dir=tmp_path / "bad_topology", proposal_budget=2)

    report["topology_mutated"] = False
    report["normalized"] = False
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="normalized must be true"):
        run_and_write_proposal_batch_report(edge_utility_report=report_path, output_dir=tmp_path / "bad_normalized", proposal_budget=2)

    report["normalized"] = True
    report["bounded_active_adjacency"] = False
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="bounded_active_adjacency must be true"):
        run_and_write_proposal_batch_report(edge_utility_report=report_path, output_dir=tmp_path / "bad_bounded", proposal_budget=2)


def test_proposal_batch_rejects_invalid_utility_rows(tmp_path):
    report_path = _build_edge_utility_fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(Path(report["edge_utility_path"]))
    rows[0]["utility_score"] = 1.5
    _write_jsonl(Path(report["edge_utility_path"]), rows)
    with pytest.raises(ValueError, match="utility_score must be in \\[0,1\\]"):
        run_and_write_proposal_batch_report(edge_utility_report=report_path, output_dir=tmp_path / "bad_utility", proposal_budget=2)

    rows[0]["utility_score"] = 0.5
    rows[0]["archive_priority"] = -0.1
    _write_jsonl(Path(report["edge_utility_path"]), rows)
    with pytest.raises(ValueError, match="archive_priority must be in \\[0,1\\]"):
        run_and_write_proposal_batch_report(edge_utility_report=report_path, output_dir=tmp_path / "bad_archive", proposal_budget=2)


def test_proposal_batch_cli_writes_report_and_prints_summary(tmp_path, capsys):
    report_path = _build_edge_utility_fixture(tmp_path)
    out = tmp_path / "proposal_cli"

    rc = main([
        "--edge-utility-report",
        str(report_path),
        "--output-dir",
        str(out),
        "--proposal-budget",
        "3",
        "--archive-priority-threshold",
        "0.50",
        "--downweight-priority-threshold",
        "0.35",
        "--keep-utility-threshold",
        "0.75",
        "--overwrite",
    ])

    assert rc == 0
    assert (out / PROPOSAL_BATCH_REPORT_FILENAME).exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "proposal_batch_generation_ok"
    assert printed["proposal_batch_report"] == str(out / PROPOSAL_BATCH_REPORT_FILENAME)
