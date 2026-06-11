from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.adaptive_rewire_contract import (
    CONTRACT_MANIFEST_FILENAME,
    REQUIRED_ARTIFACT_FILENAMES,
    load_adaptive_rewire_contract_manifest,
    main,
    run_and_write_adaptive_rewire_contract,
    validate_adaptive_rewire_contract_manifest,
)
from src.qwen_rewire_apply import run_and_write_accepted_candidate_manifest
from src.qwen_rewire_candidate_promotion import (
    CANDIDATE_NEXT_PRIOR_MANIFEST_FILENAME,
    run_and_write_candidate_promotion_report,
)
from src.qwen_rewire_next_prior import NEXT_SPARSE_PRIOR_MANIFEST_FILENAME, run_and_write_next_sparse_prior_handoff
from src.qwen_rewire_recursive_bootstrap import (
    RECURSIVE_BOOTSTRAP_MANIFEST_FILENAME,
    load_recursive_bootstrap_manifest,
    run_and_write_recursive_bootstrap_handoff,
)
from tests.test_qwen_rewire_apply import _build_accepted_search


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_recursive_bootstrap_manifest(tmp_path: Path) -> Path:
    _eval_out, _trace_out, search_out = _build_accepted_search(tmp_path)
    apply_out = tmp_path / "apply"
    run_and_write_accepted_candidate_manifest(rewire_search_dir=search_out, output_dir=apply_out)
    promotion_out = tmp_path / "candidate_promotion"
    run_and_write_candidate_promotion_report(
        accepted_candidate_dir=apply_out,
        output_dir=promotion_out,
        k=1,
        runtime_repeats=1,
        max_runtime_seconds=10.0,
        max_peak_memory_bytes=128 * 1024 * 1024,
        max_cuda_peak_memory_bytes=128 * 1024 * 1024,
        device="torch_cpu",
    )
    next_prior_out = tmp_path / "next_prior"
    run_and_write_next_sparse_prior_handoff(
        candidate_next_prior_manifest=promotion_out / CANDIDATE_NEXT_PRIOR_MANIFEST_FILENAME,
        output_dir=next_prior_out,
    )
    bootstrap_out = tmp_path / "recursive_bootstrap"
    run_and_write_recursive_bootstrap_handoff(
        next_sparse_prior_manifest=next_prior_out / NEXT_SPARSE_PRIOR_MANIFEST_FILENAME,
        output_dir=bootstrap_out,
        cycle_index=1,
        k=1,
        feature_dim=8,
        steps=1,
        seeds=[0, 1, 2],
        device="torch_cpu",
    )
    return bootstrap_out / RECURSIVE_BOOTSTRAP_MANIFEST_FILENAME


def _mutated_bootstrap_manifest(tmp_path: Path, source_manifest_path: Path, **updates) -> Path:
    source = load_recursive_bootstrap_manifest(source_manifest_path.parent)
    source.update(updates)
    path = tmp_path / "mutated" / RECURSIVE_BOOTSTRAP_MANIFEST_FILENAME
    _write_json(path, source)
    return path


def _run_contract(source: Path, out: Path) -> dict:
    return run_and_write_adaptive_rewire_contract(
        recursive_bootstrap_manifest=source,
        output_dir=out,
        student_id="qwen_style_tiny",
        experiment_id="v26_p9_cycle_001_contract",
        edge_budget=32,
        proposal_budget=8,
        closure_horizon=1,
        acceptance_policy="measured_gated_noop_contract",
        old_domain_regression_budget=0.0,
        rollback_enabled=True,
    )


def test_adaptive_rewire_contract_happy_path_writes_plan_artifacts(tmp_path):
    bootstrap_manifest = _build_recursive_bootstrap_manifest(tmp_path)
    out = tmp_path / "adaptive_rewire_contract"

    summary = _run_contract(bootstrap_manifest, out)

    assert summary["status"] == "adaptive_rewire_contract_run_ok"
    for filename in REQUIRED_ARTIFACT_FILENAMES.values():
        assert (out / filename).exists()
    assert (out / CONTRACT_MANIFEST_FILENAME).exists()
    initial = json.loads((out / "initial_adjacency.json").read_text(encoding="utf-8"))
    final = json.loads((out / "final_adjacency.json").read_text(encoding="utf-8"))
    assert final == initial
    manifest = load_adaptive_rewire_contract_manifest(out)
    validation = validate_adaptive_rewire_contract_manifest(manifest)
    assert validation["canonical_artifacts_ready"] is True
    assert validation["required_artifacts_present"] is True
    assert manifest["contract_only"] is True
    assert manifest["topology_mutated"] is False
    assert manifest["final_equals_initial"] is True
    assert manifest["bounded_active_adjacency"] is True


def test_adaptive_rewire_contract_rejects_non_ready_bootstrap(tmp_path):
    bootstrap_manifest = _build_recursive_bootstrap_manifest(tmp_path)
    bad_edge = _mutated_bootstrap_manifest(tmp_path, bootstrap_manifest, edge_trace_ready=False)
    with pytest.raises(ValueError, match="edge_trace_ready must be true"):
        _run_contract(bad_edge, tmp_path / "bad_edge")

    bad_recursive = _mutated_bootstrap_manifest(tmp_path, bootstrap_manifest, recursive_seed_ready=False)
    with pytest.raises(ValueError, match="recursive_seed_ready must be true"):
        _run_contract(bad_recursive, tmp_path / "bad_recursive")


def test_adaptive_rewire_contract_rejects_mutation_flags(tmp_path):
    bootstrap_manifest = _build_recursive_bootstrap_manifest(tmp_path)
    bad_base = _mutated_bootstrap_manifest(tmp_path, bootstrap_manifest, base_topology_mutated=True)
    with pytest.raises(ValueError, match="base_topology_mutated must be false"):
        _run_contract(bad_base, tmp_path / "bad_base")

    bad_active = _mutated_bootstrap_manifest(tmp_path, bootstrap_manifest, active_topology_mutated=True)
    with pytest.raises(ValueError, match="active_topology_mutated must be false"):
        _run_contract(bad_active, tmp_path / "bad_active")

    bad_applied = _mutated_bootstrap_manifest(tmp_path, bootstrap_manifest, proposal_applied_to_base=True)
    with pytest.raises(ValueError, match="proposal_applied_to_base must be false"):
        _run_contract(bad_applied, tmp_path / "bad_applied")


def test_adaptive_rewire_contract_rejects_unsafe_runtime_flags(tmp_path):
    bootstrap_manifest = _build_recursive_bootstrap_manifest(tmp_path)
    bad_teacher = _mutated_bootstrap_manifest(tmp_path, bootstrap_manifest, teacher_checkpoint_loaded=True)
    with pytest.raises(ValueError, match="teacher_checkpoint_loaded must be false"):
        _run_contract(bad_teacher, tmp_path / "bad_teacher")

    bad_runtime = _mutated_bootstrap_manifest(tmp_path, bootstrap_manifest, teacher_inference_runtime_required=True)
    with pytest.raises(ValueError, match="teacher_inference_runtime_required must be false"):
        _run_contract(bad_runtime, tmp_path / "bad_runtime")

    bad_payload = _mutated_bootstrap_manifest(tmp_path, bootstrap_manifest, raw_weight_payload_in_graph=True)
    with pytest.raises(ValueError, match="raw_weight_payload_in_graph must be false"):
        _run_contract(bad_payload, tmp_path / "bad_payload")


def test_adaptive_rewire_contract_cli_writes_manifest_and_prints_summary(tmp_path, capsys):
    bootstrap_manifest = _build_recursive_bootstrap_manifest(tmp_path)
    out = tmp_path / "adaptive_rewire_contract_cli"

    rc = main([
        "--recursive-bootstrap-manifest",
        str(bootstrap_manifest),
        "--output-dir",
        str(out),
        "--student-id",
        "qwen_style_tiny",
        "--experiment-id",
        "v26_p9_cycle_001_contract",
        "--edge-budget",
        "32",
        "--proposal-budget",
        "8",
        "--closure-horizon",
        "1",
        "--acceptance-policy",
        "measured_gated_noop_contract",
        "--old-domain-regression-budget",
        "0.0",
        "--rollback-enabled",
        "--overwrite",
    ])

    assert rc == 0
    assert (out / CONTRACT_MANIFEST_FILENAME).exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "adaptive_rewire_contract_run_ok"
    assert printed["adaptive_rewire_contract_manifest"] == str(out / CONTRACT_MANIFEST_FILENAME)
