from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.adaptive_rewire_bounded_graph import (
    BOUNDED_CANDIDATE_ADJACENCY_FILENAME,
    BOUNDED_GRAPH_REPORT_FILENAME,
    PROTECTED_EDGES_FILENAME,
    load_bounded_graph_report,
    main,
    run_and_write_bounded_graph_report,
    validate_bounded_graph_report,
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


def _source_adjacency() -> dict:
    return {
        "schema_version": "qwen_selected_adjacency.v1",
        "adjacency_name": "qwen_topk_k1_v26_candidate",
        "selection_policy": "test_fixture",
        "source": "G_0",
        "bounded": True,
        "k": 1,
        "edge_count": 2,
        "node_count": 4,
        "edges": [
            {
                "edge_id": "edge_a",
                "src_id": "node_a",
                "dst_id": "node_b",
                "relation": "attn",
                "weight": 1.0,
                "score_name": "normalized_frobenius",
            },
            {
                "edge_id": "edge_b",
                "src_id": "node_c",
                "dst_id": "node_d",
                "relation": "mlp",
                "weight": 0.5,
                "score_name": "normalized_frobenius",
            },
        ],
    }


def _proposal(
    index: int,
    operation: str,
    edge_id: str,
    source_node: str,
    target_node: str,
    edge_type: str,
    *,
    closure_critical: bool = False,
    reason_codes: list[str] | None = None,
    proposed_delta: dict | None = None,
    source_provenance: dict | None = None,
) -> dict:
    return {
        "schema_version": "adaptive_rewire_proposal.row.v1",
        "student_id": "qwen_style_tiny",
        "experiment_id": "v26_p11_proposal_batch",
        "cycle_index": 1,
        "proposal_index": index,
        "operation": operation,
        "edge_id": edge_id,
        "source_node": source_node,
        "target_node": target_node,
        "edge_type": edge_type,
        "utility_score": 0.8 if operation == "keep" else 0.2,
        "archive_priority": 0.1 if operation == "keep" else 0.9,
        "add_priority": 0.0,
        "closure_critical_flag": closure_critical,
        "reason_codes": reason_codes or ["fixture"],
        "source_provenance": source_provenance or {},
        "proposed_delta": proposed_delta or {"topology_mutated": False},
        "rollback_data": {"operation": "restore_edge_state", "edge_id": edge_id},
        "auto_accepted": False,
        "accepted": False,
        "rejected": False,
        "topology_mutated": False,
    }


def _build_proposal_fixture(tmp_path: Path, proposals: list[dict]) -> Path:
    contract_dir = tmp_path / "contract"
    _write_json(contract_dir / "initial_adjacency.json", _source_adjacency())
    contract_manifest = {
        "schema_version": "adaptive_rewire_contract.v1",
        "status": "adaptive_rewire_contract_ok",
        "output_dir": str(contract_dir),
        "artifacts": {"initial_adjacency": "initial_adjacency.json"},
    }
    contract_manifest_path = contract_dir / "adaptive_rewire_contract_manifest.json"
    _write_json(contract_manifest_path, contract_manifest)
    edge_utility_dir = tmp_path / "edge_utility"
    edge_utility_report = {
        "schema_version": "adaptive_rewire_edge_utility.v1",
        "status": "edge_utility_report_ok",
        "source_contract_manifest": str(contract_manifest_path),
        "output_dir": str(edge_utility_dir),
        "edge_utility_path": str(edge_utility_dir / "edge_utility.jsonl"),
        "edge_count": 2,
        "active_edge_count": 2,
        "trace_row_count": 6,
        "normalized": True,
        "topology_mutated": False,
        "bounded_active_adjacency": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "next_stage": "proposal_batch_generator",
    }
    edge_report_path = edge_utility_dir / "edge_utility_report.json"
    _write_json(edge_report_path, edge_utility_report)
    _write_jsonl(edge_utility_dir / "edge_utility.jsonl", [])
    proposal_dir = tmp_path / "proposal_batch"
    proposal_path = proposal_dir / "proposal_batch.jsonl"
    _write_jsonl(proposal_path, proposals)
    report = {
        "schema_version": "adaptive_rewire_proposal_batch.v1",
        "status": "proposal_batch_report_ok",
        "source_edge_utility_report": str(edge_report_path),
        "output_dir": str(proposal_dir),
        "proposal_batch_path": str(proposal_path),
        "proposal_budget": max(1, len(proposals)),
        "proposal_count": len(proposals),
        "archive_proposal_count": sum(1 for row in proposals if row["operation"] == "archive"),
        "downweight_proposal_count": sum(1 for row in proposals if row["operation"] == "downweight"),
        "keep_proposal_count": sum(1 for row in proposals if row["operation"] == "keep"),
        "add_proposal_count": sum(1 for row in proposals if row["operation"] == "add"),
        "tombstone_proposal_count": sum(1 for row in proposals if row["operation"] == "tombstone"),
        "candidate_pool_available": False,
        "bounded_proposal_count": True,
        "auto_accepted": False,
        "topology_mutated": False,
        "bounded_active_adjacency": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "next_stage": "bounded_active_graph_enforcement",
    }
    report_path = proposal_dir / "proposal_batch_report.json"
    _write_json(report_path, report)
    return report_path


def test_bounded_graph_happy_path_writes_candidate_and_protection_outputs(tmp_path):
    report_path = _build_proposal_fixture(tmp_path, [
        _proposal(0, "keep", "edge_a", "node_a", "node_b", "attn", reason_codes=["high_utility_keep"]),
        _proposal(1, "downweight", "edge_b", "node_c", "node_d", "mlp"),
    ])
    out = tmp_path / "bounded_graph"

    summary = run_and_write_bounded_graph_report(
        proposal_batch_report=report_path,
        output_dir=out,
        edge_budget=2,
        max_out_degree_limit=1,
    )

    assert summary["status"] == "bounded_graph_enforcement_ok"
    assert (out / BOUNDED_GRAPH_REPORT_FILENAME).exists()
    assert (out / BOUNDED_CANDIDATE_ADJACENCY_FILENAME).exists()
    assert (out / PROTECTED_EDGES_FILENAME).exists()
    report = load_bounded_graph_report(out)
    assert validate_bounded_graph_report(report)["bounded_active_adjacency"] is True
    assert report["bounded_edge_count"] is True
    assert report["bounded_out_degree"] is True
    assert report["topology_mutated"] is False
    assert report["rewrites_accepted"] is False


def test_bounded_graph_budget_fail_closed_rejects_excess_adds(tmp_path):
    add_edge = {
        "edge": {
            "edge_id": "edge_new",
            "src_id": "node_e",
            "dst_id": "node_f",
            "relation": "attn",
            "weight": 0.1,
            "score_name": "adaptive_rewire_candidate",
        }
    }
    report_path = _build_proposal_fixture(tmp_path, [
        _proposal(
            0,
            "add",
            "edge_new",
            "node_e",
            "node_f",
            "attn",
            proposed_delta=add_edge,
            source_provenance={"candidate_pool_available": True},
        )
    ])
    out = tmp_path / "bounded_graph_budget"

    run_and_write_bounded_graph_report(proposal_batch_report=report_path, output_dir=out, edge_budget=2, max_out_degree_limit=1)

    report = load_bounded_graph_report(out)
    assert report["candidate_edge_count"] <= report["edge_budget"]
    assert report["rejected_for_budget_count"] > 0


def test_bounded_graph_protected_edge_fail_closed(tmp_path):
    report_path = _build_proposal_fixture(tmp_path, [
        _proposal(0, "keep", "edge_a", "node_a", "node_b", "attn", reason_codes=["high_utility_keep"]),
        _proposal(1, "archive", "edge_a", "node_a", "node_b", "attn"),
    ])
    out = tmp_path / "bounded_graph_protected"

    run_and_write_bounded_graph_report(proposal_batch_report=report_path, output_dir=out, edge_budget=2, max_out_degree_limit=1)

    report = load_bounded_graph_report(out)
    candidate = json.loads((out / BOUNDED_CANDIDATE_ADJACENCY_FILENAME).read_text(encoding="utf-8"))
    edge_ids = {edge["edge_id"] for edge in candidate["edges"]}
    assert "edge_a" in edge_ids
    assert report["rejected_for_protection_count"] > 0


def test_bounded_graph_rejects_unsafe_proposal_report(tmp_path):
    report_path = _build_proposal_fixture(tmp_path, [
        _proposal(0, "keep", "edge_a", "node_a", "node_b", "attn"),
    ])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["auto_accepted"] = True
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="auto_accepted must be false"):
        run_and_write_bounded_graph_report(proposal_batch_report=report_path, output_dir=tmp_path / "bad_auto", edge_budget=2)

    report["auto_accepted"] = False
    report["topology_mutated"] = True
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="topology_mutated must be false"):
        run_and_write_bounded_graph_report(proposal_batch_report=report_path, output_dir=tmp_path / "bad_topology", edge_budget=2)

    report["topology_mutated"] = False
    report["bounded_active_adjacency"] = False
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="bounded_active_adjacency must be true"):
        run_and_write_bounded_graph_report(proposal_batch_report=report_path, output_dir=tmp_path / "bad_bounded", edge_budget=2)


def test_bounded_graph_cli_writes_report_and_prints_summary(tmp_path, capsys):
    report_path = _build_proposal_fixture(tmp_path, [
        _proposal(0, "keep", "edge_a", "node_a", "node_b", "attn"),
    ])
    out = tmp_path / "bounded_graph_cli"

    rc = main([
        "--proposal-batch-report",
        str(report_path),
        "--output-dir",
        str(out),
        "--edge-budget",
        "2",
        "--max-out-degree-limit",
        "1",
        "--overwrite",
    ])

    assert rc == 0
    assert (out / BOUNDED_GRAPH_REPORT_FILENAME).exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "bounded_graph_enforcement_ok"
    assert printed["bounded_graph_report"] == str(out / BOUNDED_GRAPH_REPORT_FILENAME)
