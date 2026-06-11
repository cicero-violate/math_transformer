from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from src.qwen_graph_prior_eval import run_graph_prior_eval
from src.qwen_kl_distillation_eval import (
    evaluate_kl_against_frozen_logits,
    kl_divergence,
    project_features_to_logits,
    run_and_write_kl_eval_report,
)
from src.qwen_logit_distillation_targets import write_frozen_logit_distillation_targets
from src.qwen_weight_graph import QwenWeightGraphCompiler, build_tensor_manifest_from_directory, write_weight_graph_artifacts


def _make_safetensors_bytes(tensors: dict[str, np.ndarray]) -> bytes:
    header: dict = {"__metadata__": {"format": "pt"}}
    offset = 0
    data_parts: list[bytes] = []
    for name, arr in tensors.items():
        arr_f32 = arr.astype(np.float32)
        data = arr_f32.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        data_parts.append(data)
        offset += len(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(data_parts)


def _energy_proxy_tensors() -> dict[str, np.ndarray]:
    rng = np.random.RandomState(77)
    tensors = {}
    for li in range(2):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors[f"model.layers.{li}.self_attn.{proj}.weight"] = rng.randn(8, 8).astype(np.float32)
        for proj in ("gate_proj", "up_proj"):
            tensors[f"model.layers.{li}.mlp.{proj}.weight"] = rng.randn(16, 8).astype(np.float32)
        tensors[f"model.layers.{li}.mlp.down_proj.weight"] = rng.randn(8, 16).astype(np.float32)
    tensors["model.norm.weight"] = rng.randn(8).astype(np.float32)
    return tensors


def _loader(tensors: dict[str, np.ndarray]):
    def load(spec):
        arr = tensors.get(spec.name)
        return arr.astype(np.float32) if arr is not None else None

    return load


def _compile_g0(tmp_path: Path) -> Path:
    tensors = _energy_proxy_tensors()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(_make_safetensors_bytes(tensors))
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "kl_eval_test")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=2, source_model="kl_eval_test")
    result = compiler.compile(specs, _loader(tensors), config_hash=config_hash, index_hash=index_hash)
    out = tmp_path / "g0"
    write_weight_graph_artifacts(result, out)
    return out


def _build_eval_output(tmp_path: Path) -> Path:
    g0 = _compile_g0(tmp_path)
    out = tmp_path / "prior_eval"
    run_graph_prior_eval(
        source_weight_graph_dir=g0,
        output_dir=out,
        k_values=[1, 2],
        random_seeds=[0, 1],
        quality_mode="energy_capture",
    )
    return out


def _build_eval_and_logit_targets(tmp_path: Path) -> tuple[Path, Path]:
    out = _build_eval_output(tmp_path)
    targets = tmp_path / "logit_targets"
    write_frozen_logit_distillation_targets(targets, vocab_size=16, seeds=[0, 1, 2], temperature=1.0)
    return out, targets


def test_evaluate_kl_against_frozen_logits(tmp_path):
    out, targets = _build_eval_and_logit_targets(tmp_path)
    report = evaluate_kl_against_frozen_logits(out, targets, k=1)
    assert report["status"] == "kl_distillation_eval_ok"
    assert report["student_training_started"] is False
    assert report["teacher_checkpoint_loaded"] is False
    assert report["teacher_inference_runtime_required"] is False
    assert report["teacher_distillation_started"] is False
    assert report["kl_training_started"] is False
    assert report["raw_weight_payload_in_graph"] is False
    assert report["bounded_active_adjacency"] is True
    assert report["promotion_eligible"] is False
    assert report["vocab_size"] == 16
    assert report["row_count"] == 3
    assert report["kl_mean"] >= 0.0
    assert report["kl_min"] >= 0.0
    assert report["kl_max"] >= report["kl_min"]
    assert report["finite"] is True
    assert len(report["rows"]) == 3
    for row in report["rows"]:
        assert {"kl", "cross_entropy", "entropy_teacher", "student_logits_checksum", "student_probabilities_checksum"}.issubset(row)


def test_kl_eval_is_deterministic(tmp_path):
    out, targets = _build_eval_and_logit_targets(tmp_path)
    first = evaluate_kl_against_frozen_logits(out, targets, k=1, projection_seed=0)
    second = evaluate_kl_against_frozen_logits(out, targets, k=1, projection_seed=0)
    assert first["kl_mean"] == second["kl_mean"]
    assert first["kl_min"] == second["kl_min"]
    assert first["kl_max"] == second["kl_max"]
    assert first["cross_entropy_mean"] == second["cross_entropy_mean"]
    assert first["entropy_teacher_mean"] == second["entropy_teacher_mean"]
    assert first["rows"] == second["rows"]


def test_kl_eval_changes_with_projection_seed(tmp_path):
    out, targets = _build_eval_and_logit_targets(tmp_path)
    first = evaluate_kl_against_frozen_logits(out, targets, k=1, projection_seed=0)
    second = evaluate_kl_against_frozen_logits(out, targets, k=1, projection_seed=1)
    first_checksums = [row["student_logits_checksum"] for row in first["rows"]]
    second_checksums = [row["student_logits_checksum"] for row in second["rows"]]
    assert first_checksums != second_checksums
    assert first["rows"] != second["rows"]


def test_kl_eval_writes_report(tmp_path):
    out, targets = _build_eval_and_logit_targets(tmp_path)
    report_path = tmp_path / "reports" / "kl_eval.json"
    report = run_and_write_kl_eval_report(out, targets, report_path, k=1)
    assert report_path.exists()
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded == report


def test_kl_divergence_simple_case():
    metrics = kl_divergence([1.0, 0.0], [0.5, 0.5])
    assert metrics["kl"] == pytest.approx(math.log(2.0), abs=1e-10)
    assert metrics["finite"] is True


def test_project_features_to_logits_is_deterministic():
    features = {"b": [0.25, -0.5], "a": [1.0, 2.0]}
    first = project_features_to_logits(features, vocab_size=4, seed=7)
    second = project_features_to_logits(features, vocab_size=4, seed=7)
    assert first == second
    assert len(first) == 4
    assert all(math.isfinite(value) for value in first)


def test_kl_eval_rejects_bad_temperature_and_steps(tmp_path):
    out, targets = _build_eval_and_logit_targets(tmp_path)
    with pytest.raises(ValueError, match="temperature must be > 0"):
        evaluate_kl_against_frozen_logits(out, targets, k=1, temperature=0.0)
    with pytest.raises(ValueError, match="temperature must be finite"):
        evaluate_kl_against_frozen_logits(out, targets, k=1, temperature=float("nan"))
    with pytest.raises(ValueError, match="steps must be >= 1"):
        evaluate_kl_against_frozen_logits(out, targets, k=1, steps=0)
    with pytest.raises(ValueError, match="feature_dim must be >= 1"):
        evaluate_kl_against_frozen_logits(out, targets, k=1, feature_dim=0)


def test_kl_eval_rejects_invalid_logit_targets(tmp_path):
    out, targets = _build_eval_and_logit_targets(tmp_path)
    manifest_path = targets / "frozen_logit_targets_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["kl_training_started"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="kl_training_started"):
        evaluate_kl_against_frozen_logits(out, targets, k=1)
