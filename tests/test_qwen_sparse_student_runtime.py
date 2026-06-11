from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.qwen_graph_prior_eval import run_graph_prior_eval
from src.qwen_sparse_student_runtime import compute_feature_mse, run_fixed_topology_forward, run_fixed_topology_loss_dry_run
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
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "runtime_test")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=2, source_model="runtime_test")
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


def test_fixed_topology_forward_runs_from_v25_handoff(tmp_path):
    out = _build_eval_output(tmp_path)
    summary = run_fixed_topology_forward(out, k=1, feature_dim=8, steps=1, seed=123)
    assert summary["status"] == "fixed_topology_forward_ok"
    assert summary["student_training_started"] is False
    assert summary["teacher_checkpoint_loaded"] is False
    assert summary["raw_weight_payload_in_graph"] is False
    assert summary["bounded_active_adjacency"] is True
    assert summary["adjacency_name"] == "qwen_topk_k1"
    assert summary["k"] == 1
    assert summary["feature_dim"] == 8
    assert summary["steps"] == 1
    assert summary["node_count"] > 0
    assert summary["edge_count"] > 0
    assert summary["max_out_degree"] <= 1
    assert summary["finite"] is True
    assert summary["changed_node_count"] > 0
    assert summary["input_checksum"] != summary["output_checksum"]
    assert summary["ready_for_v25_distillation"] is False


def test_fixed_topology_forward_is_deterministic(tmp_path):
    out = _build_eval_output(tmp_path)
    first = run_fixed_topology_forward(out, k=1, feature_dim=8, steps=2, seed=123)
    second = run_fixed_topology_forward(out, k=1, feature_dim=8, steps=2, seed=123)
    assert first["input_checksum"] == second["input_checksum"]
    assert first["output_checksum"] == second["output_checksum"]


def test_fixed_topology_forward_changes_with_seed(tmp_path):
    out = _build_eval_output(tmp_path)
    first = run_fixed_topology_forward(out, k=1, feature_dim=8, steps=1, seed=1)
    second = run_fixed_topology_forward(out, k=1, feature_dim=8, steps=1, seed=2)
    assert first["input_checksum"] != second["input_checksum"]
    assert first["output_checksum"] != second["output_checksum"]


def test_fixed_topology_forward_selects_k2(tmp_path):
    out = _build_eval_output(tmp_path)
    summary = run_fixed_topology_forward(out, k=2, feature_dim=8, steps=1, seed=123)
    assert summary["adjacency_name"] == "qwen_topk_k2"
    assert summary["k"] == 2
    assert summary["max_out_degree"] <= 2


def test_fixed_topology_forward_rejects_bad_dimensions(tmp_path):
    out = _build_eval_output(tmp_path)
    with pytest.raises(ValueError, match="feature_dim must be >= 1"):
        run_fixed_topology_forward(out, k=1, feature_dim=0)
    with pytest.raises(ValueError, match="steps must be >= 1"):
        run_fixed_topology_forward(out, k=1, steps=0)


def test_fixed_topology_loss_dry_run_scores_forward_output(tmp_path):
    out = _build_eval_output(tmp_path)
    summary = run_fixed_topology_loss_dry_run(
        out,
        k=1,
        feature_dim=8,
        steps=1,
        seed=123,
        target_mode="identity",
    )
    assert summary["status"] == "fixed_topology_loss_dry_run_ok"
    assert summary["student_training_started"] is False
    assert summary["teacher_checkpoint_loaded"] is False
    assert summary["teacher_distillation_started"] is False
    assert summary["raw_weight_payload_in_graph"] is False
    assert summary["bounded_active_adjacency"] is True
    assert summary["adjacency_name"] == "qwen_topk_k1"
    assert summary["k"] == 1
    assert summary["feature_dim"] == 8
    assert summary["steps"] == 1
    assert summary["target_mode"] == "identity"
    assert summary["loss_mse"] >= 0.0
    assert summary["loss_l1"] >= 0.0
    assert summary["finite"] is True
    assert summary["input_checksum"] != summary["output_checksum"]
    assert summary["target_checksum"] == summary["input_checksum"]
    assert summary["ready_for_teacher_distillation"] is False
    assert summary["promotion_eligible"] is False


def test_loss_dry_run_is_deterministic(tmp_path):
    out = _build_eval_output(tmp_path)
    first = run_fixed_topology_loss_dry_run(out, k=1, feature_dim=8, steps=2, seed=123)
    second = run_fixed_topology_loss_dry_run(out, k=1, feature_dim=8, steps=2, seed=123)
    assert first["loss_mse"] == second["loss_mse"]
    assert first["loss_l1"] == second["loss_l1"]
    assert first["input_checksum"] == second["input_checksum"]
    assert first["output_checksum"] == second["output_checksum"]
    assert first["target_checksum"] == second["target_checksum"]


def test_loss_dry_run_target_modes_change_target_checksum(tmp_path):
    out = _build_eval_output(tmp_path)
    identity = run_fixed_topology_loss_dry_run(out, k=1, feature_dim=8, steps=1, seed=123, target_mode="identity")
    zero = run_fixed_topology_loss_dry_run(out, k=1, feature_dim=8, steps=1, seed=123, target_mode="zero")
    scaled = run_fixed_topology_loss_dry_run(
        out,
        k=1,
        feature_dim=8,
        steps=1,
        seed=123,
        target_mode="scaled_identity",
        target_scale=0.5,
    )
    assert identity["target_checksum"] == identity["input_checksum"]
    assert zero["target_checksum"] != identity["target_checksum"]
    assert scaled["target_checksum"] != identity["target_checksum"]
    assert scaled["target_checksum"] != zero["target_checksum"]
    assert identity["finite"] is True
    assert zero["finite"] is True
    assert scaled["finite"] is True


def test_compute_feature_mse_rejects_mismatched_nodes():
    with pytest.raises(ValueError, match="node ids must match"):
        compute_feature_mse({"a": [1.0]}, {"b": [1.0]})


def test_loss_dry_run_rejects_bad_target_mode_and_scale(tmp_path):
    out = _build_eval_output(tmp_path)
    with pytest.raises(ValueError, match="target_mode"):
        run_fixed_topology_loss_dry_run(out, k=1, target_mode="teacher")
    with pytest.raises(ValueError, match="target_scale must be finite"):
        run_fixed_topology_loss_dry_run(out, k=1, target_mode="scaled_identity", target_scale=float("nan"))
