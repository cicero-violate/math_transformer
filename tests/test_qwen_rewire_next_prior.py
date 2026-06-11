from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.qwen_rewire_apply import run_and_write_accepted_candidate_manifest
from src.qwen_rewire_candidate_promotion import (
    CANDIDATE_NEXT_PRIOR_MANIFEST_FILENAME,
    load_candidate_next_prior_manifest,
    run_and_write_candidate_promotion_report,
)
from src.qwen_rewire_next_prior import (
    NEXT_SPARSE_PRIOR_ADJACENCY_SUMMARY_FILENAME,
    NEXT_SPARSE_PRIOR_EVAL_DIRNAME,
    NEXT_SPARSE_PRIOR_MANIFEST_FILENAME,
    load_next_sparse_prior_manifest,
    main,
    run_and_write_next_sparse_prior_handoff,
    validate_next_sparse_prior_manifest,
)
from src.qwen_sparse_student_handoff import load_selected_adjacency, validate_selected_adjacency
from tests.test_qwen_rewire_apply import _build_accepted_search


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_promoted_candidate_manifest(tmp_path: Path) -> Path:
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
    return promotion_out / CANDIDATE_NEXT_PRIOR_MANIFEST_FILENAME


def _mutated_candidate_manifest(tmp_path: Path, source_manifest_path: Path, **updates) -> Path:
    source = load_candidate_next_prior_manifest(source_manifest_path.parent)
    source.update(updates)
    path = tmp_path / "mutated" / CANDIDATE_NEXT_PRIOR_MANIFEST_FILENAME
    _write_json(path, source)
    return path


def test_next_prior_happy_path_copies_promoted_candidate_without_mutating_source(tmp_path):
    candidate_manifest = _build_promoted_candidate_manifest(tmp_path)
    candidate = load_candidate_next_prior_manifest(candidate_manifest.parent)
    source_eval_dir = Path(candidate["applied_candidate_eval_dir"])
    source_adjacency_before = load_selected_adjacency(source_eval_dir, adjacency_name=candidate["candidate_adjacency_name"])
    out = tmp_path / "next_prior"

    summary = run_and_write_next_sparse_prior_handoff(
        candidate_next_prior_manifest=candidate_manifest,
        output_dir=out,
    )

    assert summary["status"] == "next_sparse_prior_handoff_ok"
    assert (out / NEXT_SPARSE_PRIOR_MANIFEST_FILENAME).exists()
    assert (out / NEXT_SPARSE_PRIOR_EVAL_DIRNAME).exists()
    assert (out / NEXT_SPARSE_PRIOR_ADJACENCY_SUMMARY_FILENAME).exists()
    manifest = load_next_sparse_prior_manifest(out)
    validation = validate_next_sparse_prior_manifest(manifest)
    assert validation["recursive_seed_ready"] is True
    assert validation["next_cycle_input_ready"] is True
    assert manifest["recursive_seed_ready"] is True
    assert manifest["next_cycle_input_ready"] is True
    assert manifest["bounded_active_adjacency"] is True
    assert manifest["base_topology_mutated"] is False
    assert manifest["active_topology_mutated"] is False
    assert manifest["proposal_applied_to_base"] is False
    copied = load_selected_adjacency(
        manifest["next_sparse_prior_eval_dir"],
        adjacency_name=manifest["next_sparse_prior_adjacency_name"],
    )
    assert validate_selected_adjacency(copied)["bounded"] is True
    assert copied == source_adjacency_before
    assert load_selected_adjacency(source_eval_dir, adjacency_name=candidate["candidate_adjacency_name"]) == source_adjacency_before


def test_next_prior_rejects_unpromoted_candidate(tmp_path):
    candidate_manifest = _build_promoted_candidate_manifest(tmp_path)
    bad_manifest = _mutated_candidate_manifest(
        tmp_path,
        candidate_manifest,
        candidate_promoted=False,
        decision="candidate_not_promoted",
        promotion_eligible=False,
    )
    with pytest.raises(ValueError, match="candidate_promoted must be true"):
        run_and_write_next_sparse_prior_handoff(
            candidate_next_prior_manifest=bad_manifest,
            output_dir=tmp_path / "next_prior_bad",
        )


def test_next_prior_rejects_failed_gate(tmp_path):
    candidate_manifest = _build_promoted_candidate_manifest(tmp_path)
    bad_quality = _mutated_candidate_manifest(tmp_path, candidate_manifest, quality_ok=False)
    with pytest.raises(ValueError, match="quality_ok must be true"):
        run_and_write_next_sparse_prior_handoff(
            candidate_next_prior_manifest=bad_quality,
            output_dir=tmp_path / "next_prior_bad_quality",
        )

    bad_runtime = _mutated_candidate_manifest(tmp_path, candidate_manifest, runtime_ok=False)
    with pytest.raises(ValueError, match="runtime_ok must be true"):
        run_and_write_next_sparse_prior_handoff(
            candidate_next_prior_manifest=bad_runtime,
            output_dir=tmp_path / "next_prior_bad_runtime",
        )


def test_next_prior_rejects_mutation_flags(tmp_path):
    candidate_manifest = _build_promoted_candidate_manifest(tmp_path)
    bad_base = _mutated_candidate_manifest(tmp_path, candidate_manifest, base_topology_mutated=True)
    with pytest.raises(ValueError, match="base_topology_mutated must be false"):
        run_and_write_next_sparse_prior_handoff(
            candidate_next_prior_manifest=bad_base,
            output_dir=tmp_path / "next_prior_bad_base",
        )

    bad_active = _mutated_candidate_manifest(tmp_path, candidate_manifest, active_topology_mutated=True)
    with pytest.raises(ValueError, match="active_topology_mutated must be false"):
        run_and_write_next_sparse_prior_handoff(
            candidate_next_prior_manifest=bad_active,
            output_dir=tmp_path / "next_prior_bad_active",
        )


def test_next_prior_cli_writes_manifest_and_prints_summary(tmp_path, capsys):
    candidate_manifest = _build_promoted_candidate_manifest(tmp_path)
    out = tmp_path / "next_prior_cli"

    rc = main([
        "--candidate-next-prior-manifest",
        str(candidate_manifest),
        "--output-dir",
        str(out),
        "--overwrite",
    ])

    assert rc == 0
    assert (out / NEXT_SPARSE_PRIOR_MANIFEST_FILENAME).exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "next_sparse_prior_handoff_ok"
    assert printed["next_sparse_prior_manifest"] == str(out / NEXT_SPARSE_PRIOR_MANIFEST_FILENAME)
