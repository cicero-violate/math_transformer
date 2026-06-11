from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.qwen_distillation_harness import run_fixed_topology_distillation_harness
from src.qwen_distillation_measured_gates import (
    build_measured_distillation_gate_report,
    load_measured_distillation_gate_report,
    run_and_write_measured_distillation_gate_report,
    validate_measured_distillation_gate_report,
)
from src.qwen_distillation_promotion import decide_distillation_promotion, run_and_write_distillation_promotion_decision
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
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "distillation_measured_gate_test")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=2, source_model="distillation_measured_gate_test")
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


def test_measured_gates_pass_and_enable_p12_promotion(tmp_path):
    harness_out = _build_harness_output(tmp_path)
    report = build_measured_distillation_gate_report(
        harness_out,
        runtime_repeats=2,
        max_runtime_seconds=10.0,
        max_peak_memory_bytes=128 * 1024 * 1024,
    )
    assert report["status"] == "distillation_gate_report_ok"
    assert report["quality_gate"]["available"] is True
    assert report["quality_gate"]["proxy_only"] is False
    assert report["quality_gate"]["task_quality_available"] is True
    assert report["quality_gate"]["quality_ok"] is True
    assert report["quality_gate"]["kl_final"] <= report["quality_gate"]["kl_initial"]
    assert report["runtime_gate"]["available"] is True
    assert report["runtime_gate"]["runtime_ok"] is True
    assert report["memory_gate"]["available"] is True
    assert report["memory_gate"]["memory_ok"] is True
    assert report["safety_gate"]["safety_ok"] is True
    assert report["promotion_report"]["promote"] is True
    assert validate_measured_distillation_gate_report(report)["promote_ready"] is True

    decision = decide_distillation_promotion(report)
    assert decision["promote"] is True
    assert decision["decision"] == "promoted"
    assert decision["missing_or_failed_gates"] == []


def test_measured_gate_report_writes_and_promotion_writes(tmp_path):
    harness_out = _build_harness_output(tmp_path)
    report_path = harness_out / "distillation_measured_gate_report.json"
    report = run_and_write_measured_distillation_gate_report(
        harness_out,
        report_path,
        runtime_repeats=1,
        max_runtime_seconds=10.0,
        max_peak_memory_bytes=128 * 1024 * 1024,
    )
    assert report_path.exists()
    assert load_measured_distillation_gate_report(report_path) == report
    decision_path = harness_out / "distillation_measured_promotion_decision.json"
    decision = run_and_write_distillation_promotion_decision(report_path, decision_path)
    assert decision_path.exists()
    assert decision["promote"] is True


def test_measured_runtime_gate_can_fail_without_promotion(tmp_path):
    harness_out = _build_harness_output(tmp_path)
    report = build_measured_distillation_gate_report(
        harness_out,
        runtime_repeats=1,
        max_runtime_seconds=0.0,
        max_peak_memory_bytes=128 * 1024 * 1024,
    )
    assert report["runtime_gate"]["runtime_ok"] is False
    assert report["memory_gate"]["memory_ok"] is True
    assert report["promotion_report"]["promote"] is False
    decision = decide_distillation_promotion(report)
    assert decision["promote"] is False
    assert "runtime_ok" in decision["missing_or_failed_gates"]


def test_measured_memory_gate_can_fail_without_promotion(tmp_path):
    harness_out = _build_harness_output(tmp_path)
    report = build_measured_distillation_gate_report(
        harness_out,
        runtime_repeats=1,
        max_runtime_seconds=10.0,
        max_peak_memory_bytes=0,
    )
    assert report["runtime_gate"]["runtime_ok"] is True
    assert report["memory_gate"]["memory_ok"] is False
    assert report["promotion_report"]["promote"] is False
    decision = decide_distillation_promotion(report)
    assert decision["promote"] is False
    assert "memory_ok" in decision["missing_or_failed_gates"]


def test_measured_quality_gate_can_fail_threshold_without_promotion(tmp_path):
    harness_out = _build_harness_output(tmp_path)
    report = build_measured_distillation_gate_report(
        harness_out,
        runtime_repeats=1,
        max_runtime_seconds=10.0,
        max_peak_memory_bytes=128 * 1024 * 1024,
        min_kl_relative_reduction=1.0,
    )
    assert report["quality_gate"]["quality_ok"] is False
    assert report["promotion_report"]["promote"] is False
    decision = decide_distillation_promotion(report)
    assert decision["promote"] is False
    assert "quality_ok" in decision["missing_or_failed_gates"]


def test_measured_gates_record_device_and_cuda_measurement_contract(tmp_path):
    harness_out = _build_harness_output(tmp_path)
    report = build_measured_distillation_gate_report(
        harness_out,
        runtime_repeats=1,
        max_runtime_seconds=10.0,
        max_peak_memory_bytes=128 * 1024 * 1024,
        device="torch_cpu",
    )
    protocol = report["measurement_protocol"]
    assert protocol["device"] == "torch_cpu"
    assert protocol["requested_device"] == "torch_cpu"
    assert protocol["resolved_device"] == "cpu"
    assert protocol["runtime_backend"] == "torch"
    assert protocol["torch_available"] is True
    assert protocol["cuda_measurement_available"] is False
    assert protocol["cuda_runtime_protocol"] == "unavailable"
    assert protocol["cuda_memory_protocol"] == "unavailable"
    assert report["runtime_gate"]["cuda_duration_seconds"] == []
    assert report["runtime_gate"]["cuda_duration_median_seconds"] is None
    assert report["memory_gate"]["host_memory_ok"] is True
    assert report["memory_gate"]["cuda_memory_ok"] is None
    assert report["memory_gate"]["cuda_peak_bytes"] == []
    assert report["memory_gate"]["cuda_peak_max_bytes"] is None
    assert validate_measured_distillation_gate_report(report)["promote_ready"] is True


def test_run_and_write_measured_gates_forwards_device_override(tmp_path):
    harness_out = _build_harness_output(tmp_path)
    report_path = harness_out / "torch_cpu_measured_gate_report.json"
    report = run_and_write_measured_distillation_gate_report(
        harness_out,
        report_path,
        runtime_repeats=1,
        max_runtime_seconds=10.0,
        max_peak_memory_bytes=128 * 1024 * 1024,
        device="torch_cpu",
    )
    assert report_path.exists()
    assert report["measurement_protocol"]["device"] == "torch_cpu"
    assert report["measurement_protocol"]["runtime_backend"] == "torch"


def test_measured_gates_reject_bad_args(tmp_path):
    harness_out = _build_harness_output(tmp_path)
    with pytest.raises(ValueError, match="runtime_repeats"):
        build_measured_distillation_gate_report(harness_out, runtime_repeats=0)
    with pytest.raises(ValueError, match="max_runtime_seconds"):
        build_measured_distillation_gate_report(harness_out, max_runtime_seconds=-1.0)
    with pytest.raises(ValueError, match="max_peak_memory_bytes"):
        build_measured_distillation_gate_report(harness_out, max_peak_memory_bytes=-1)
    with pytest.raises(ValueError, match="min_kl_relative_reduction"):
        build_measured_distillation_gate_report(harness_out, min_kl_relative_reduction=-0.1)
