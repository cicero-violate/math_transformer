from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from src.qwen_logit_distillation_targets import (
    iter_frozen_logit_target_rows,
    softmax,
    validate_frozen_logit_distillation_targets,
    write_frozen_logit_distillation_targets,
)


def _read_manifest(target_out: Path) -> dict:
    return json.loads((target_out / "frozen_logit_targets_manifest.json").read_text(encoding="utf-8"))


def _write_manifest(target_out: Path, manifest: dict) -> None:
    (target_out / "frozen_logit_targets_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rewrite_rows_and_refresh_manifest(target_out: Path, rows: list[dict]) -> None:
    rows_path = target_out / "frozen_logit_targets.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = _read_manifest(target_out)
    manifest["target_rows_sha256"] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    _write_manifest(target_out, manifest)


def test_write_and_validate_frozen_logit_targets(tmp_path):
    target_out = tmp_path / "logit_targets"
    manifest = write_frozen_logit_distillation_targets(target_out, vocab_size=16, seeds=[0, 1, 2], temperature=1.0)
    assert (target_out / "frozen_logit_targets_manifest.json").exists()
    assert (target_out / "frozen_logit_targets.jsonl").exists()
    assert manifest["status"] == "frozen_logit_targets_ready"
    assert manifest["target_type"] == "logits"
    assert manifest["teacher_checkpoint_loaded_at_runtime"] is False
    assert manifest["teacher_inference_runtime_required"] is False
    assert manifest["raw_weight_payload_in_graph"] is False
    assert manifest["student_training_started"] is False
    assert manifest["kl_training_started"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["vocab_size"] == 16
    assert manifest["row_count"] == 3
    summary = validate_frozen_logit_distillation_targets(target_out)
    assert summary["status"] == "frozen_logit_targets_valid"
    assert summary["row_count"] == 3
    assert summary["vocab_size"] == 16


def test_frozen_logit_targets_are_deterministic(tmp_path):
    first_dir = tmp_path / "targets_a"
    second_dir = tmp_path / "targets_b"
    first = write_frozen_logit_distillation_targets(first_dir, vocab_size=16, seeds=[0, 1, 2], temperature=1.0)
    second = write_frozen_logit_distillation_targets(second_dir, vocab_size=16, seeds=[0, 1, 2], temperature=1.0)
    assert first["target_rows_sha256"] == second["target_rows_sha256"]
    first_checksums = [
        (row["logits_checksum"], row["probabilities_checksum"])
        for row in iter_frozen_logit_target_rows(first_dir)
    ]
    second_checksums = [
        (row["logits_checksum"], row["probabilities_checksum"])
        for row in iter_frozen_logit_target_rows(second_dir)
    ]
    assert first_checksums == second_checksums


def test_frozen_logit_targets_change_with_seed(tmp_path):
    first = write_frozen_logit_distillation_targets(tmp_path / "targets_a", vocab_size=16, seeds=[0], temperature=1.0)
    second = write_frozen_logit_distillation_targets(tmp_path / "targets_b", vocab_size=16, seeds=[1], temperature=1.0)
    assert first["target_rows_sha256"] != second["target_rows_sha256"]


def test_softmax_probabilities_sum_to_one():
    probabilities = softmax([1.0, 2.0, 3.0])
    assert all(probability >= 0.0 for probability in probabilities)
    assert math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12)


def test_validate_rejects_bad_probability_sum(tmp_path):
    target_out = tmp_path / "targets"
    write_frozen_logit_distillation_targets(target_out, vocab_size=16, seeds=[0, 1, 2], temperature=1.0)
    rows = list(iter_frozen_logit_target_rows(target_out))
    rows[0]["probabilities"][0] += 0.1
    _rewrite_rows_and_refresh_manifest(target_out, rows)
    with pytest.raises(ValueError, match="probabilities"):
        validate_frozen_logit_distillation_targets(target_out)


def test_validate_rejects_bad_checksum(tmp_path):
    target_out = tmp_path / "targets"
    write_frozen_logit_distillation_targets(target_out, vocab_size=16, seeds=[0, 1, 2], temperature=1.0)
    rows = list(iter_frozen_logit_target_rows(target_out))
    rows[0]["logits"][0] += 1.0
    _rewrite_rows_and_refresh_manifest(target_out, rows)
    with pytest.raises(ValueError, match="logits_checksum mismatch"):
        validate_frozen_logit_distillation_targets(target_out)


def test_validate_rejects_runtime_teacher_true(tmp_path):
    target_out = tmp_path / "targets"
    write_frozen_logit_distillation_targets(target_out, vocab_size=16, seeds=[0, 1, 2], temperature=1.0)
    manifest = _read_manifest(target_out)
    manifest["teacher_checkpoint_loaded_at_runtime"] = True
    _write_manifest(target_out, manifest)
    with pytest.raises(ValueError, match="teacher_checkpoint_loaded_at_runtime"):
        validate_frozen_logit_distillation_targets(target_out)


def test_validate_rejects_forbidden_payload_key(tmp_path):
    target_out = tmp_path / "targets"
    write_frozen_logit_distillation_targets(target_out, vocab_size=16, seeds=[0, 1, 2], temperature=1.0)
    rows = list(iter_frozen_logit_target_rows(target_out))
    rows[0]["metadata"]["raw_weight_payload"] = [1, 2, 3]
    _rewrite_rows_and_refresh_manifest(target_out, rows)
    with pytest.raises(ValueError, match="raw tensor payload field forbidden"):
        validate_frozen_logit_distillation_targets(target_out)


def test_validate_rejects_bad_temperature_or_vocab(tmp_path):
    with pytest.raises(ValueError, match="vocab_size must be >= 2"):
        write_frozen_logit_distillation_targets(tmp_path / "bad_vocab", vocab_size=1)
    with pytest.raises(ValueError, match="temperature must be > 0"):
        write_frozen_logit_distillation_targets(tmp_path / "bad_temp", temperature=0.0)
    with pytest.raises(ValueError, match="temperature must be finite"):
        write_frozen_logit_distillation_targets(tmp_path / "bad_temp_nan", temperature=float("nan"))
