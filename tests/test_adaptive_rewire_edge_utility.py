from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.adaptive_rewire_contract import REQUIRED_ARTIFACT_FILENAMES
from src.adaptive_rewire_edge_utility import (
    EDGE_UTILITY_FILENAME,
    EDGE_UTILITY_REPORT_FILENAME,
    load_adaptive_rewire_edge_utility_report,
    main,
    run_and_write_edge_utility_report,
    validate_edge_utility_report,
)
from src.qwen_edge_trace import EDGE_TRACE_REPORT_FILENAME


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


def _adjacency() -> dict:
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
                "edge_id": "edge_b",
                "src_id": "node_b",
                "dst_id": "node_c",
                "relation": "mlp",
                "weight": 0.25,
                "score_name": "normalized_frobenius",
            },
            {
                "edge_id": "edge_a",
                "src_id": "node_a",
                "dst_id": "node_d",
                "relation": "attn",
                "weight": 0.75,
                "score_name": "normalized_frobenius",
            },
        ],
    }


def _edge_trace_report() -> dict:
    return {
        "schema_version": "qwen_edge_trace.v1",
        "status": "edge_trace_ok",
        "student_training_started": False,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "teacher_distillation_started": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "topology_mutated": False,
        "eval_output_dir": "cycle_001_seed_eval",
        "adjacency_name": "qwen_topk_k1_v26_candidate",
        "k": 1,
        "edge_count": 2,
        "node_count": 4,
        "max_out_degree": 1,
        "feature_dim": 8,
        "steps": 1,
        "seeds": [0, 1, 2],
        "row_count": 6,
        "expected_row_count": 6,
        "device": "torch_cpu",
        "edge_utility_summary": {
            "schema_version": "qwen_edge_utility_summary.v1",
            "status": "edge_utility_summary_ok",
            "edge_count": 2,
            "row_count": 6,
            "finite": True,
            "ranked_edges": [
                {
                    "edge_id": "edge_a",
                    "src_id": "node_a",
                    "dst_id": "node_d",
                    "relation": "attn",
                    "score_name": "normalized_frobenius",
                    "weight": 0.75,
                    "normalized_weight": 1.0,
                    "trace_count": 3,
                    "used_count": 3,
                    "utility_score": 2.0,
                    "message_l1_mean": 2.0,
                    "message_l1_max": 3.0,
                    "message_l2_mean": 1.0,
                    "message_l2_max": 1.5,
                    "src_l1_mean": 2.0,
                    "dst_delta_l1_mean": 2.0,
                    "finite": True,
                },
                {
                    "edge_id": "edge_b",
                    "src_id": "node_b",
                    "dst_id": "node_c",
                    "relation": "mlp",
                    "score_name": "normalized_frobenius",
                    "weight": 0.25,
                    "normalized_weight": 1.0,
                    "trace_count": 3,
                    "used_count": 3,
                    "utility_score": 1.0,
                    "message_l1_mean": 1.0,
                    "message_l1_max": 1.5,
                    "message_l2_mean": 0.5,
                    "message_l2_max": 1.0,
                    "src_l1_mean": 1.0,
                    "dst_delta_l1_mean": 1.0,
                    "finite": True,
                },
            ],
        },
        "artifacts": {
            "edge_trace_report": "edge_trace_report.json",
            "edge_trace_rows": "edge_trace.jsonl",
            "edge_utility_summary": "edge_utility_summary.json",
        },
        "finite": True,
    }


def _build_contract_fixture(tmp_path: Path, *, with_edge_trace: bool = True) -> Path:
    contract_dir = tmp_path / "adaptive_rewire_contract"
    contract_dir.mkdir(parents=True)
    adjacency = _adjacency()
    _write_json(contract_dir / "initial_adjacency.json", adjacency)
    _write_json(contract_dir / "final_adjacency.json", adjacency)
    for key, filename in REQUIRED_ARTIFACT_FILENAMES.items():
        path = contract_dir / filename
        if path.exists():
            continue
        if filename.endswith(".jsonl"):
            _write_jsonl(path, [{"schema_version": "fixture.noop.v1", "stream": key, "noop": True}])
        else:
            _write_json(path, {"schema_version": "fixture.noop.v1", "status": "ok"})

    bootstrap_dir = tmp_path / "bootstrap"
    next_prior = tmp_path / "next_prior" / "next_sparse_prior_manifest.json"
    next_prior.parent.mkdir(parents=True)
    _write_json(next_prior, {"schema_version": "fixture.v1", "status": "exists"})
    seed_eval = bootstrap_dir / "cycle_001_seed_eval"
    edge_trace = bootstrap_dir / "cycle_001_edge_trace"
    source_eval = tmp_path / "source_next_prior_eval"
    seed_eval.mkdir(parents=True)
    source_eval.mkdir(parents=True)
    edge_trace.mkdir(parents=True)
    if with_edge_trace:
        _write_json(edge_trace / EDGE_TRACE_REPORT_FILENAME, _edge_trace_report())
    bootstrap_manifest = {
        "schema_version": "qwen_rewire_recursive_bootstrap.v1",
        "status": "recursive_bootstrap_manifest_ok",
        "cycle_index": 1,
        "source_next_sparse_prior_manifest": str(next_prior),
        "source_next_sparse_prior_eval_dir": str(source_eval),
        "cycle_seed_eval_dir": str(seed_eval),
        "cycle_edge_trace_dir": str(edge_trace),
        "cycle_seed_adjacency_name": "qwen_topk_k1_v26_candidate",
        "selected_candidate_index": 4,
        "selected_candidate_policy": "deterministic_random",
        "selected_candidate_kl_delta": -0.03418879930089158,
        "recursive_seed_ready": True,
        "next_cycle_input_ready": True,
        "edge_trace_ready": True,
        "bounded_active_adjacency": True,
        "base_topology_mutated": False,
        "active_topology_mutated": False,
        "proposal_applied_to_base": False,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "cycle_seed_adjacency_summary": {
            "adjacency_name": "qwen_topk_k1_v26_candidate",
            "bounded": True,
            "edge_count": 2,
            "k": 1,
            "max_out_degree": 1,
            "node_count": 4,
            "selection_policy": "test_fixture",
            "source": "G_0",
        },
        "edge_trace_summary": {
            "edge_count": 2,
            "row_count": 6,
            "adjacency_name": "qwen_topk_k1_v26_candidate",
            "topology_mutated": False,
        },
        "artifacts": {},
    }
    _write_json(bootstrap_dir / "recursive_bootstrap_manifest.json", bootstrap_manifest)
    manifest = {
        "schema_version": "adaptive_rewire_contract.v1",
        "status": "adaptive_rewire_contract_ok",
        "source_recursive_bootstrap_manifest": str(bootstrap_dir / "recursive_bootstrap_manifest.json"),
        "output_dir": str(contract_dir),
        "student_id": "qwen_style_tiny",
        "experiment_id": "v26_p9_cycle_001_contract",
        "cycle_index": 1,
        "contract_only": True,
        "topology_mutated": False,
        "final_equals_initial": True,
        "edge_trace_ready": True,
        "canonical_artifacts_ready": True,
        "required_artifacts_present": True,
        "bounded_active_adjacency": True,
        "teacher_checkpoint_loaded": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "next_stage": "edge_utility_aggregator",
        "artifacts": dict(REQUIRED_ARTIFACT_FILENAMES),
    }
    contract_manifest = contract_dir / "adaptive_rewire_contract_manifest.json"
    _write_json(contract_manifest, manifest)
    return contract_manifest


def test_edge_utility_happy_path_writes_normalized_rows(tmp_path):
    contract_manifest = _build_contract_fixture(tmp_path)
    out = tmp_path / "edge_utility"

    summary = run_and_write_edge_utility_report(contract_manifest=contract_manifest, output_dir=out)

    assert summary["status"] == "edge_utility_aggregation_ok"
    assert (out / EDGE_UTILITY_FILENAME).exists()
    assert (out / EDGE_UTILITY_REPORT_FILENAME).exists()
    rows = _read_jsonl(out / EDGE_UTILITY_FILENAME)
    assert [(row["source_node"], row["target_node"], row["edge_type"]) for row in rows] == [
        ("node_a", "node_d", "attn"),
        ("node_b", "node_c", "mlp"),
    ]
    for row in rows:
        assert 0.0 <= row["utility_score"] <= 1.0
        assert 0.0 <= row["archive_priority"] <= 1.0
        assert 0.0 <= row["add_priority"] <= 1.0
        assert row["topology_mutated"] is False if "topology_mutated" in row else True
    report = load_adaptive_rewire_edge_utility_report(out)
    assert validate_edge_utility_report(report)["normalized"] is True
    assert report["topology_mutated"] is False


def test_edge_utility_missing_optional_gradient_stats_does_not_crash(tmp_path):
    contract_manifest = _build_contract_fixture(tmp_path)
    out = tmp_path / "edge_utility_missing_optional"

    run_and_write_edge_utility_report(contract_manifest=contract_manifest, output_dir=out)

    report = load_adaptive_rewire_edge_utility_report(out)
    assert report["missing_gradient_stats"] is True
    rows = _read_jsonl(out / EDGE_UTILITY_FILENAME)
    assert all(row["gradient_norm"] is None for row in rows)


def test_edge_utility_rejects_unsafe_contract(tmp_path):
    contract_manifest = _build_contract_fixture(tmp_path)
    manifest = json.loads(contract_manifest.read_text(encoding="utf-8"))
    manifest["topology_mutated"] = True
    _write_json(contract_manifest, manifest)
    with pytest.raises(ValueError, match="topology_mutated must be false"):
        run_and_write_edge_utility_report(contract_manifest=contract_manifest, output_dir=tmp_path / "bad_topology")

    manifest["topology_mutated"] = False
    manifest["bounded_active_adjacency"] = False
    _write_json(contract_manifest, manifest)
    with pytest.raises(ValueError, match="bounded_active_adjacency must be true"):
        run_and_write_edge_utility_report(contract_manifest=contract_manifest, output_dir=tmp_path / "bad_bounded")


def test_edge_utility_rejects_missing_required_artifacts_and_trace(tmp_path):
    contract_manifest = _build_contract_fixture(tmp_path)
    manifest = json.loads(contract_manifest.read_text(encoding="utf-8"))
    (Path(manifest["output_dir"]) / "initial_adjacency.json").unlink()
    with pytest.raises(ValueError, match="initial_adjacency.json"):
        run_and_write_edge_utility_report(contract_manifest=contract_manifest, output_dir=tmp_path / "bad_initial")

    missing_trace_manifest = _build_contract_fixture(tmp_path / "missing_trace", with_edge_trace=False)
    with pytest.raises(FileNotFoundError, match="edge trace source missing"):
        run_and_write_edge_utility_report(contract_manifest=missing_trace_manifest, output_dir=tmp_path / "bad_trace")


def test_edge_utility_cli_writes_report_and_prints_summary(tmp_path, capsys):
    contract_manifest = _build_contract_fixture(tmp_path)
    out = tmp_path / "edge_utility_cli"

    rc = main([
        "--contract-manifest",
        str(contract_manifest),
        "--output-dir",
        str(out),
        "--overwrite",
    ])

    assert rc == 0
    assert (out / EDGE_UTILITY_REPORT_FILENAME).exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "edge_utility_aggregation_ok"
    assert printed["edge_utility_report"] == str(out / EDGE_UTILITY_REPORT_FILENAME)
