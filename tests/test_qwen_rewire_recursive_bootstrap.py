from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.qwen_rewire_apply import run_and_write_accepted_candidate_manifest
from src.qwen_rewire_candidate_promotion import (
    CANDIDATE_NEXT_PRIOR_MANIFEST_FILENAME,
    run_and_write_candidate_promotion_report,
)
from src.qwen_rewire_next_prior import (
    NEXT_SPARSE_PRIOR_MANIFEST_FILENAME,
    load_next_sparse_prior_manifest,
    run_and_write_next_sparse_prior_handoff,
)
from src.qwen_rewire_recursive_bootstrap import (
    CYCLE_EDGE_TRACE_DIRNAME_TEMPLATE,
    CYCLE_SEED_ADJACENCY_SUMMARY_FILENAME_TEMPLATE,
    CYCLE_SEED_EVAL_DIRNAME_TEMPLATE,
    RECURSIVE_BOOTSTRAP_MANIFEST_FILENAME,
    load_recursive_bootstrap_manifest,
    main,
    run_and_write_recursive_bootstrap_handoff,
    validate_recursive_bootstrap_manifest,
)
from src.qwen_sparse_student_handoff import load_selected_adjacency, validate_selected_adjacency
from tests.test_qwen_rewire_apply import _build_accepted_search


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_next_sparse_prior_manifest(tmp_path: Path) -> Path:
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
    return next_prior_out / NEXT_SPARSE_PRIOR_MANIFEST_FILENAME


def _mutated_next_prior_manifest(tmp_path: Path, source_manifest_path: Path, **updates) -> Path:
    source = load_next_sparse_prior_manifest(source_manifest_path.parent)
    source.update(updates)
    path = tmp_path / "mutated" / NEXT_SPARSE_PRIOR_MANIFEST_FILENAME
    _write_json(path, source)
    return path


def test_recursive_bootstrap_happy_path_writes_cycle_seed_and_edge_trace(tmp_path):
    next_prior_manifest = _build_next_sparse_prior_manifest(tmp_path)
    next_prior = load_next_sparse_prior_manifest(next_prior_manifest.parent)
    source_eval_dir = Path(next_prior["next_sparse_prior_eval_dir"])
    source_adjacency_before = load_selected_adjacency(
        source_eval_dir,
        adjacency_name=next_prior["next_sparse_prior_adjacency_name"],
    )
    out = tmp_path / "recursive_bootstrap"

    summary = run_and_write_recursive_bootstrap_handoff(
        next_sparse_prior_manifest=next_prior_manifest,
        output_dir=out,
        cycle_index=1,
        k=1,
        feature_dim=8,
        steps=1,
        seeds=[0, 1, 2],
        device="torch_cpu",
    )

    assert summary["status"] == "recursive_bootstrap_handoff_ok"
    assert (out / RECURSIVE_BOOTSTRAP_MANIFEST_FILENAME).exists()
    assert (out / CYCLE_SEED_EVAL_DIRNAME_TEMPLATE.format(cycle=1)).exists()
    assert (out / CYCLE_EDGE_TRACE_DIRNAME_TEMPLATE.format(cycle=1)).exists()
    assert (out / CYCLE_SEED_ADJACENCY_SUMMARY_FILENAME_TEMPLATE.format(cycle=1)).exists()

    manifest = load_recursive_bootstrap_manifest(out)
    validation = validate_recursive_bootstrap_manifest(manifest)
    assert validation["recursive_seed_ready"] is True
    assert validation["next_cycle_input_ready"] is True
    assert validation["edge_trace_ready"] is True
    assert manifest["recursive_seed_ready"] is True
    assert manifest["next_cycle_input_ready"] is True
    assert manifest["edge_trace_ready"] is True
    assert manifest["bounded_active_adjacency"] is True
    assert manifest["base_topology_mutated"] is False
    assert manifest["active_topology_mutated"] is False
    assert manifest["proposal_applied_to_base"] is False
    copied = load_selected_adjacency(
        manifest["cycle_seed_eval_dir"],
        adjacency_name=manifest["cycle_seed_adjacency_name"],
    )
    assert validate_selected_adjacency(copied)["bounded"] is True
    assert copied == source_adjacency_before
    assert load_selected_adjacency(
        source_eval_dir,
        adjacency_name=next_prior["next_sparse_prior_adjacency_name"],
    ) == source_adjacency_before


def test_recursive_bootstrap_rejects_non_ready_seed(tmp_path):
    next_prior_manifest = _build_next_sparse_prior_manifest(tmp_path)
    bad_recursive = _mutated_next_prior_manifest(tmp_path, next_prior_manifest, recursive_seed_ready=False)
    with pytest.raises(ValueError, match="recursive_seed_ready must be true"):
        run_and_write_recursive_bootstrap_handoff(
            next_sparse_prior_manifest=bad_recursive,
            output_dir=tmp_path / "bad_recursive",
        )

    bad_cycle = _mutated_next_prior_manifest(tmp_path, next_prior_manifest, next_cycle_input_ready=False)
    with pytest.raises(ValueError, match="next_cycle_input_ready must be true"):
        run_and_write_recursive_bootstrap_handoff(
            next_sparse_prior_manifest=bad_cycle,
            output_dir=tmp_path / "bad_cycle",
        )


def test_recursive_bootstrap_rejects_mutation_flags(tmp_path):
    next_prior_manifest = _build_next_sparse_prior_manifest(tmp_path)
    bad_base = _mutated_next_prior_manifest(tmp_path, next_prior_manifest, base_topology_mutated=True)
    with pytest.raises(ValueError, match="base_topology_mutated must be false"):
        run_and_write_recursive_bootstrap_handoff(
            next_sparse_prior_manifest=bad_base,
            output_dir=tmp_path / "bad_base",
        )

    bad_active = _mutated_next_prior_manifest(tmp_path, next_prior_manifest, active_topology_mutated=True)
    with pytest.raises(ValueError, match="active_topology_mutated must be false"):
        run_and_write_recursive_bootstrap_handoff(
            next_sparse_prior_manifest=bad_active,
            output_dir=tmp_path / "bad_active",
        )

    bad_applied = _mutated_next_prior_manifest(tmp_path, next_prior_manifest, proposal_applied_to_base=True)
    with pytest.raises(ValueError, match="proposal_applied_to_base must be false"):
        run_and_write_recursive_bootstrap_handoff(
            next_sparse_prior_manifest=bad_applied,
            output_dir=tmp_path / "bad_applied",
        )


def test_recursive_bootstrap_rejects_unsafe_runtime_flags(tmp_path):
    next_prior_manifest = _build_next_sparse_prior_manifest(tmp_path)
    bad_teacher = _mutated_next_prior_manifest(tmp_path, next_prior_manifest, teacher_checkpoint_loaded=True)
    with pytest.raises(ValueError, match="teacher_checkpoint_loaded must be false"):
        run_and_write_recursive_bootstrap_handoff(
            next_sparse_prior_manifest=bad_teacher,
            output_dir=tmp_path / "bad_teacher",
        )

    bad_runtime = _mutated_next_prior_manifest(tmp_path, next_prior_manifest, teacher_inference_runtime_required=True)
    with pytest.raises(ValueError, match="teacher_inference_runtime_required must be false"):
        run_and_write_recursive_bootstrap_handoff(
            next_sparse_prior_manifest=bad_runtime,
            output_dir=tmp_path / "bad_runtime",
        )

    bad_payload = _mutated_next_prior_manifest(tmp_path, next_prior_manifest, raw_weight_payload_in_graph=True)
    with pytest.raises(ValueError, match="raw_weight_payload_in_graph must be false"):
        run_and_write_recursive_bootstrap_handoff(
            next_sparse_prior_manifest=bad_payload,
            output_dir=tmp_path / "bad_payload",
        )


def test_recursive_bootstrap_cli_writes_manifest_and_prints_summary(tmp_path, capsys):
    next_prior_manifest = _build_next_sparse_prior_manifest(tmp_path)
    out = tmp_path / "recursive_bootstrap_cli"

    rc = main([
        "--next-sparse-prior-manifest",
        str(next_prior_manifest),
        "--output-dir",
        str(out),
        "--cycle-index",
        "1",
        "--k",
        "1",
        "--feature-dim",
        "8",
        "--steps",
        "1",
        "--seeds",
        "0,1,2",
        "--device",
        "torch_cpu",
        "--overwrite",
    ])

    assert rc == 0
    assert (out / RECURSIVE_BOOTSTRAP_MANIFEST_FILENAME).exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "recursive_bootstrap_handoff_ok"
    assert printed["recursive_bootstrap_manifest"] == str(out / RECURSIVE_BOOTSTRAP_MANIFEST_FILENAME)
