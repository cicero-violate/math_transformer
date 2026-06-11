from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.qwen_rewire_apply import run_and_write_accepted_candidate_manifest
from src.qwen_rewire_candidate_promotion import (
    CANDIDATE_DISTILLATION_HARNESS_REPORT_FILENAME,
    CANDIDATE_MEASURED_GATE_REPORT_FILENAME,
    CANDIDATE_NEXT_PRIOR_MANIFEST_FILENAME,
    CANDIDATE_PROMOTION_DECISION_FILENAME,
    load_candidate_next_prior_manifest,
    main,
    run_and_write_candidate_promotion_report,
    validate_candidate_next_prior_manifest,
)
from src.qwen_sparse_student_handoff import load_selected_adjacency, validate_selected_adjacency
from tests.test_qwen_rewire_apply import _build_accepted_search


def _build_accepted_candidate_apply(tmp_path: Path) -> tuple[Path, Path]:
    eval_out, _trace_out, search_out = _build_accepted_search(tmp_path)
    apply_out = tmp_path / "apply"
    run_and_write_accepted_candidate_manifest(rewire_search_dir=search_out, output_dir=apply_out)
    return eval_out, apply_out


def test_candidate_promotion_happy_path_writes_next_prior_manifest_without_mutating_base(tmp_path):
    eval_out, apply_out = _build_accepted_candidate_apply(tmp_path)
    base_before = load_selected_adjacency(eval_out, k=1)
    out = tmp_path / "candidate_promotion"

    summary = run_and_write_candidate_promotion_report(
        accepted_candidate_dir=apply_out,
        output_dir=out,
        k=1,
        vocab_size=16,
        target_seeds=[0, 1, 2],
        feature_dim=8,
        forward_steps=1,
        train_steps=5,
        lr=0.1,
        runtime_repeats=1,
        max_runtime_seconds=10.0,
        max_peak_memory_bytes=128 * 1024 * 1024,
        max_cuda_peak_memory_bytes=128 * 1024 * 1024,
        device="torch_cpu",
    )

    assert summary["status"] == "candidate_promotion_pipeline_ok"
    assert summary["decision"] == "candidate_promoted_as_next_prior"
    assert summary["candidate_promoted"] is True
    for filename in (
        CANDIDATE_DISTILLATION_HARNESS_REPORT_FILENAME,
        CANDIDATE_MEASURED_GATE_REPORT_FILENAME,
        CANDIDATE_PROMOTION_DECISION_FILENAME,
        CANDIDATE_NEXT_PRIOR_MANIFEST_FILENAME,
    ):
        assert (out / filename).exists()

    manifest = load_candidate_next_prior_manifest(out)
    validation = validate_candidate_next_prior_manifest(manifest)
    assert validation["candidate_promoted"] is True
    assert manifest["promotion_eligible"] is True
    assert manifest["bounded_active_adjacency"] is True
    assert manifest["base_topology_mutated"] is False
    assert manifest["active_topology_mutated"] is False
    assert manifest["proposal_applied_to_base"] is False
    candidate = load_selected_adjacency(
        manifest["applied_candidate_eval_dir"],
        adjacency_name=manifest["candidate_adjacency_name"],
    )
    assert validate_selected_adjacency(candidate)["bounded"] is True
    assert load_selected_adjacency(eval_out, k=1) == base_before


def test_candidate_promotion_rejects_when_measured_gate_fails(tmp_path):
    _eval_out, apply_out = _build_accepted_candidate_apply(tmp_path)
    out = tmp_path / "candidate_promotion_rejected"

    summary = run_and_write_candidate_promotion_report(
        accepted_candidate_dir=apply_out,
        output_dir=out,
        k=1,
        runtime_repeats=1,
        max_runtime_seconds=0.0,
        max_peak_memory_bytes=128 * 1024 * 1024,
        device="torch_cpu",
    )

    assert summary["status"] == "candidate_promotion_pipeline_ok"
    assert summary["candidate_promoted"] is False
    assert summary["decision"] == "candidate_not_promoted"
    assert summary["runtime_ok"] is False
    manifest = load_candidate_next_prior_manifest(out)
    assert manifest["candidate_promoted"] is False
    assert manifest["decision"] == "candidate_not_promoted"
    assert manifest["promotion_eligible"] is False
    assert validate_candidate_next_prior_manifest(manifest)["candidate_promoted"] is False


def test_candidate_promotion_bad_args_fail_clearly(tmp_path):
    missing = tmp_path / "missing_apply"
    with pytest.raises(FileNotFoundError, match="accepted_candidate_manifest.json"):
        run_and_write_candidate_promotion_report(accepted_candidate_dir=missing, output_dir=tmp_path / "out")

    _eval_out, apply_out = _build_accepted_candidate_apply(tmp_path)
    bad_cases = [
        ({"k": 0}, "k"),
        ({"train_steps": 0}, "train_steps"),
        ({"runtime_repeats": 0}, "runtime_repeats"),
        ({"max_peak_memory_bytes": -1}, "max_peak_memory_bytes"),
        ({"max_cuda_peak_memory_bytes": -1}, "max_cuda_peak_memory_bytes"),
    ]
    for kwargs, match in bad_cases:
        with pytest.raises(ValueError, match=match):
            run_and_write_candidate_promotion_report(
                accepted_candidate_dir=apply_out,
                output_dir=tmp_path / f"bad_{match}",
                **kwargs,
            )


def test_candidate_promotion_cli_writes_manifest_and_prints_summary(tmp_path, capsys):
    _eval_out, apply_out = _build_accepted_candidate_apply(tmp_path)
    out = tmp_path / "candidate_promotion_cli"

    rc = main([
        "--accepted-candidate-dir",
        str(apply_out),
        "--output-dir",
        str(out),
        "--k",
        "1",
        "--vocab-size",
        "16",
        "--target-seeds",
        "0,1,2",
        "--feature-dim",
        "8",
        "--forward-steps",
        "1",
        "--train-steps",
        "5",
        "--lr",
        "0.1",
        "--runtime-repeats",
        "1",
        "--max-runtime-seconds",
        "10.0",
        "--max-peak-memory-bytes",
        "134217728",
        "--max-cuda-peak-memory-bytes",
        "134217728",
        "--device",
        "torch_cpu",
    ])

    assert rc == 0
    assert (out / CANDIDATE_NEXT_PRIOR_MANIFEST_FILENAME).exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "candidate_promotion_pipeline_ok"
    assert printed["candidate_next_prior_manifest"] == str(out / CANDIDATE_NEXT_PRIOR_MANIFEST_FILENAME)
