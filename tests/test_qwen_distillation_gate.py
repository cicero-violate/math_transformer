from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.qwen_distillation_gate import (
    build_distillation_gate_report,
    load_distillation_gate_report,
    run_and_write_distillation_gate_report,
    validate_distillation_gate_report,
)
from src.qwen_distillation_harness import run_fixed_topology_distillation_harness
from src.qwen_graph_prior_eval import run_graph_prior_eval
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
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "distillation_gate_test")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=2, source_model="distillation_gate_test")
    result = compiler.compile(specs, _loader(tensors), config_hash=config_hash, index_hash=index_hash)
    out = tmp_path / "g0"
    write_weight_graph_artifacts(result, out)
    return out


def _build_harness_output(tmp_path: Path) -> Path:
    g0 = _compile_g0(tmp_path)
    prior_eval = tmp_path / "prior_eval"
    run_graph_prior_eval(
        source_weight_graph_dir=g0,
        output_dir=prior_eval,
        k_values=[1, 2],
        random_seeds=[0, 1],
        quality_mode="energy_capture",
    )
    harness_out = tmp_path / "harness"
    run_fixed_topology_distillation_harness(
        prior_eval,
        harness_out,
        k=1,
        vocab_size=16,
        target_seeds=[0, 1, 2],
        train_steps=5,
        lr=0.1,
    )
    return harness_out


def test_distillation_gate_report_blocks_promotion_until_runtime_memory_quality(tmp_path):
    harness_out = _build_harness_output(tmp_path)
    report = build_distillation_gate_report(harness_out)
    assert report["status"] == "distillation_gate_report_ok"
    assert report["student_training_started"] is True
    assert report["teacher_checkpoint_loaded"] is False
    assert report["teacher_inference_runtime_required"] is False
    assert report["teacher_distillation_started"] is False
    assert report["kl_training_started"] is True
    assert report["raw_weight_payload_in_graph"] is False
    assert report["bounded_active_adjacency"] is True
    assert report["promotion_eligible"] is False
    assert report["promotion_decision"] == "not_promoted"

    quality = report["quality_gate"]
    assert quality["available"] is True
    assert quality["proxy_only"] is True
    assert quality["task_quality_available"] is False
    assert quality["quality_ok"] is True
    assert quality["kl_after"] <= quality["kl_before"]
    assert quality["kl_delta"] >= 0.0

    assert report["runtime_gate"]["available"] is False
    assert report["runtime_gate"]["speed_ok"] is False
    assert report["memory_gate"]["available"] is False
    assert report["memory_gate"]["memory_ok"] is False
    assert report["safety_gate"]["safety_ok"] is True

    promotion = report["promotion_report"]
    assert promotion["promote"] is False
    assert promotion["quality_ok"] is True
    assert promotion["runtime_ok"] is False
    assert promotion["memory_ok"] is False
    assert promotion["safety_ok"] is True
    assert promotion["reason"] == "runtime_memory_task_quality_pending"

    summary = validate_distillation_gate_report(report)
    assert summary["status"] == "distillation_gate_report_valid"
    assert summary["promote"] is False


def test_distillation_gate_report_writes_and_loads(tmp_path):
    harness_out = _build_harness_output(tmp_path)
    report_path = harness_out / "distillation_gate_report.json"
    report = run_and_write_distillation_gate_report(harness_out, report_path)
    assert report_path.exists()
    loaded = load_distillation_gate_report(report_path)
    assert loaded == report
    assert validate_distillation_gate_report(loaded)["status"] == "distillation_gate_report_valid"


def test_distillation_gate_report_default_output_path(tmp_path):
    harness_out = _build_harness_output(tmp_path)
    report = run_and_write_distillation_gate_report(harness_out)
    default_path = harness_out / "distillation_gate_report.json"
    assert default_path.exists()
    assert load_distillation_gate_report(default_path) == report


def test_distillation_gate_rejects_missing_harness_artifact(tmp_path):
    harness_out = _build_harness_output(tmp_path)
    (harness_out / "kl_training_report.json").unlink()
    with pytest.raises(FileNotFoundError, match="kl_training_report"):
        build_distillation_gate_report(harness_out)


def test_distillation_gate_validator_rejects_promotion_true(tmp_path):
    harness_out = _build_harness_output(tmp_path)
    report = build_distillation_gate_report(harness_out)
    report["promotion_report"]["promote"] = True
    with pytest.raises(ValueError, match="must not promote"):
        validate_distillation_gate_report(report)


def test_distillation_gate_validator_rejects_runtime_or_memory_ok(tmp_path):
    harness_out = _build_harness_output(tmp_path)
    report = build_distillation_gate_report(harness_out)
    report["runtime_gate"]["available"] = True
    with pytest.raises(ValueError, match="runtime gate"):
        validate_distillation_gate_report(report)

    report = build_distillation_gate_report(harness_out)
    report["memory_gate"]["memory_ok"] = True
    with pytest.raises(ValueError, match="memory gate"):
        validate_distillation_gate_report(report)
