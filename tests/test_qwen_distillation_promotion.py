from __future__ import annotations

import json
import struct
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.qwen_distillation_gate import run_and_write_distillation_gate_report
from src.qwen_distillation_harness import run_fixed_topology_distillation_harness
from src.qwen_distillation_promotion import (
    decide_distillation_promotion,
    load_distillation_promotion_decision,
    run_and_write_distillation_promotion_decision,
    validate_distillation_promotion_decision,
)
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
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "distillation_promotion_test")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=2, source_model="distillation_promotion_test")
    result = compiler.compile(specs, _loader(tensors), config_hash=config_hash, index_hash=index_hash)
    out = tmp_path / "g0"
    write_weight_graph_artifacts(result, out)
    return out


def _build_gate_report(tmp_path: Path) -> tuple[Path, Path, dict]:
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
    gate_path = harness_out / "distillation_gate_report.json"
    gate_report = run_and_write_distillation_gate_report(harness_out, gate_path)
    return harness_out, gate_path, gate_report


def _make_future_all_real_gates_pass_report(gate_report: dict) -> dict:
    future = deepcopy(gate_report)
    future["quality_gate"]["proxy_only"] = False
    future["quality_gate"]["task_quality_available"] = True
    future["quality_gate"]["quality_ok"] = True
    future["quality_gate"]["reason"] = "real task quality gate passed"
    future["runtime_gate"]["available"] = True
    future["runtime_gate"]["runtime_ok"] = True
    future["runtime_gate"]["speed_ok"] = True
    future["runtime_gate"]["reason"] = "real runtime gate passed"
    future["memory_gate"]["available"] = True
    future["memory_gate"]["memory_ok"] = True
    future["memory_gate"]["reason"] = "real memory gate passed"
    future["promotion_report"]["quality_ok"] = True
    future["promotion_report"]["runtime_ok"] = True
    future["promotion_report"]["memory_ok"] = True
    return future


def test_distillation_promotion_blocks_current_proxy_only_gate(tmp_path):
    _harness_out, gate_path, _gate_report = _build_gate_report(tmp_path)
    decision = decide_distillation_promotion(gate_path)
    assert decision["status"] == "distillation_promotion_decision_ok"
    assert decision["promote"] is False
    assert decision["decision"] == "not_promoted"
    assert decision["reason"] == "required_real_gates_missing_or_failed"
    assert decision["teacher_checkpoint_loaded"] is False
    assert decision["teacher_inference_runtime_required"] is False
    assert decision["teacher_distillation_started"] is False
    assert decision["raw_weight_payload_in_graph"] is False
    assert decision["bounded_active_adjacency"] is True
    assert decision["promotion_eligible"] is False
    assert "task_quality_available" in decision["missing_or_failed_gates"]
    assert "quality_not_proxy_only" in decision["missing_or_failed_gates"]
    assert "runtime_gate_available" in decision["missing_or_failed_gates"]
    assert "runtime_ok" in decision["missing_or_failed_gates"]
    assert "memory_gate_available" in decision["missing_or_failed_gates"]
    assert "memory_ok" in decision["missing_or_failed_gates"]
    summary = validate_distillation_promotion_decision(decision)
    assert summary["status"] == "distillation_promotion_decision_valid"
    assert summary["promote"] is False


def test_distillation_promotion_writes_and_loads_default_path(tmp_path):
    harness_out, gate_path, _gate_report = _build_gate_report(tmp_path)
    decision = run_and_write_distillation_promotion_decision(gate_path)
    decision_path = harness_out / "distillation_promotion_decision.json"
    assert decision_path.exists()
    assert load_distillation_promotion_decision(decision_path) == decision


def test_distillation_promotion_writes_explicit_path_from_in_memory_report(tmp_path):
    _harness_out, _gate_path, gate_report = _build_gate_report(tmp_path)
    decision_path = tmp_path / "reports" / "promotion.json"
    decision = run_and_write_distillation_promotion_decision(gate_report, decision_path)
    assert decision_path.exists()
    assert load_distillation_promotion_decision(decision_path) == decision


def test_distillation_promotion_requires_output_path_for_in_memory_report(tmp_path):
    _harness_out, _gate_path, gate_report = _build_gate_report(tmp_path)
    with pytest.raises(ValueError, match="output_path is required"):
        run_and_write_distillation_promotion_decision(gate_report)


def test_distillation_promotion_allows_synthetic_future_all_real_gates_pass(tmp_path):
    _harness_out, _gate_path, gate_report = _build_gate_report(tmp_path)
    future = _make_future_all_real_gates_pass_report(gate_report)
    decision = decide_distillation_promotion(future)
    assert decision["promote"] is True
    assert decision["decision"] == "promoted"
    assert decision["reason"] == "all_real_gates_passed"
    assert decision["promotion_eligible"] is True
    assert decision["missing_or_failed_gates"] == []
    assert all(decision["required_gates"].values())
    assert validate_distillation_promotion_decision(decision)["promote"] is True


def test_distillation_promotion_validator_rejects_manual_promote_with_missing_gates(tmp_path):
    _harness_out, gate_path, _gate_report = _build_gate_report(tmp_path)
    decision = decide_distillation_promotion(gate_path)
    decision["promote"] = True
    decision["decision"] = "promoted"
    decision["promotion_eligible"] = True
    with pytest.raises(ValueError, match="promotion cannot pass"):
        validate_distillation_promotion_decision(decision)


def test_distillation_promotion_validator_rejects_proxy_only_promote(tmp_path):
    _harness_out, _gate_path, gate_report = _build_gate_report(tmp_path)
    future = _make_future_all_real_gates_pass_report(gate_report)
    decision = decide_distillation_promotion(future)
    decision["gate_summary"]["quality_proxy_only"] = True
    with pytest.raises(ValueError, match="proxy-only quality"):
        validate_distillation_promotion_decision(decision)
