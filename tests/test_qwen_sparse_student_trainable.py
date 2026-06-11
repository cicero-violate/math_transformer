from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from src.qwen_distillation_targets import write_frozen_distillation_targets
from src.qwen_graph_prior_eval import run_graph_prior_eval
from src.qwen_sparse_student_trainable import (
    scalar_affine_mse_and_grads,
    run_and_write_scalar_affine_training_report,
    run_scalar_affine_train_step,
    run_scalar_affine_training_loop,
)
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
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "trainable_test")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=2, source_model="trainable_test")
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


def _build_targets(tmp_path: Path) -> tuple[Path, Path]:
    out = _build_eval_output(tmp_path)
    targets = tmp_path / "targets"
    write_frozen_distillation_targets(
        out,
        targets,
        k=1,
        feature_dim=8,
        seeds=[0, 1, 2],
        target_mode="scaled_identity",
        target_scale=0.5,
    )
    return out, targets


def test_scalar_affine_train_step_reduces_frozen_target_loss(tmp_path):
    out, targets = _build_targets(tmp_path)
    summary = run_scalar_affine_train_step(out, targets, k=1, lr=0.01)
    assert summary["status"] == "scalar_affine_train_step_ok"
    assert summary["student_training_started"] is True
    assert summary["teacher_checkpoint_loaded"] is False
    assert summary["teacher_inference_runtime_required"] is False
    assert summary["teacher_distillation_started"] is False
    assert summary["raw_weight_payload_in_graph"] is False
    assert summary["bounded_active_adjacency"] is True
    assert summary["promotion_eligible"] is False
    assert summary["optimizer"] == "analytic_sgd_scalar_affine"
    assert summary["adjacency_name"] == "qwen_topk_k1"
    assert summary["k"] == 1
    assert summary["row_count"] == 3
    assert summary["feature_dim"] == 8
    assert summary["finite"] is True
    assert summary["loss_mse_before"] >= 0.0
    assert summary["loss_mse_after"] >= 0.0
    assert summary["loss_mse_after"] <= summary["loss_mse_before"]
    assert summary["loss_decreased"] is True
    assert summary["alpha_after"] != summary["alpha_before"] or summary["beta_after"] != summary["beta_before"]


def test_scalar_affine_train_step_is_deterministic(tmp_path):
    out, targets = _build_targets(tmp_path)
    first = run_scalar_affine_train_step(out, targets, k=1, lr=0.01)
    second = run_scalar_affine_train_step(out, targets, k=1, lr=0.01)
    assert first["alpha_after"] == second["alpha_after"]
    assert first["beta_after"] == second["beta_after"]
    assert first["grad_alpha"] == second["grad_alpha"]
    assert first["grad_beta"] == second["grad_beta"]
    assert first["loss_mse_before"] == second["loss_mse_before"]
    assert first["loss_mse_after"] == second["loss_mse_after"]


def test_scalar_affine_train_step_rejects_k_mismatch(tmp_path):
    out, targets = _build_targets(tmp_path)
    with pytest.raises(ValueError, match="requested k=2"):
        run_scalar_affine_train_step(out, targets, k=2)


def test_scalar_affine_train_step_rejects_bad_lr_and_params(tmp_path):
    out, targets = _build_targets(tmp_path)
    with pytest.raises(ValueError, match="lr must be > 0"):
        run_scalar_affine_train_step(out, targets, k=1, lr=0.0)
    with pytest.raises(ValueError, match="lr must be finite"):
        run_scalar_affine_train_step(out, targets, k=1, lr=float("nan"))
    with pytest.raises(ValueError, match="alpha must be finite"):
        run_scalar_affine_train_step(out, targets, k=1, alpha=float("nan"))
    with pytest.raises(ValueError, match="beta must be finite"):
        run_scalar_affine_train_step(out, targets, k=1, beta=float("inf"))


def test_scalar_affine_mse_and_grads_matches_simple_case():
    metrics = scalar_affine_mse_and_grads(
        {"n": [2.0]},
        {"n": [1.0]},
        alpha=1.0,
        beta=0.0,
    )
    assert metrics["loss_mse"] == pytest.approx(1.0)
    assert metrics["loss_l1"] == pytest.approx(1.0)
    assert metrics["grad_alpha"] == pytest.approx(4.0)
    assert metrics["grad_beta"] == pytest.approx(2.0)
    assert metrics["value_count"] == 1
    assert metrics["finite"] is True


def test_scalar_affine_training_loop_reduces_loss(tmp_path):
    out, targets = _build_targets(tmp_path)
    report = run_scalar_affine_training_loop(out, targets, k=1, train_steps=5, lr=0.01)
    assert report["status"] == "scalar_affine_training_loop_ok"
    assert report["student_training_started"] is True
    assert report["teacher_checkpoint_loaded"] is False
    assert report["teacher_inference_runtime_required"] is False
    assert report["teacher_distillation_started"] is False
    assert report["raw_weight_payload_in_graph"] is False
    assert report["bounded_active_adjacency"] is True
    assert report["promotion_eligible"] is False
    assert report["optimizer"] == "analytic_sgd_scalar_affine"
    assert report["train_steps"] == 5
    assert len(report["history"]) == 5
    assert report["finite"] is True
    assert report["loss_mse_final"] <= report["loss_mse_initial"]
    assert report["loss_decreased"] is True
    assert report["monotonic_nonincreasing"] is True
    assert report["alpha_final"] != report["alpha_initial"] or report["beta_final"] != report["beta_initial"]


def test_scalar_affine_training_loop_is_deterministic(tmp_path):
    out, targets = _build_targets(tmp_path)
    first = run_scalar_affine_training_loop(out, targets, k=1, train_steps=5, lr=0.01)
    second = run_scalar_affine_training_loop(out, targets, k=1, train_steps=5, lr=0.01)
    assert first["alpha_final"] == second["alpha_final"]
    assert first["beta_final"] == second["beta_final"]
    assert first["loss_mse_final"] == second["loss_mse_final"]
    assert first["loss_l1_final"] == second["loss_l1_final"]
    assert first["history"] == second["history"]


def test_scalar_affine_training_loop_writes_report(tmp_path):
    out, targets = _build_targets(tmp_path)
    report_path = tmp_path / "reports" / "scalar_affine_training.json"
    report = run_and_write_scalar_affine_training_report(out, targets, report_path, k=1, train_steps=5, lr=0.01)
    assert report_path.exists()
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded == report


def test_scalar_affine_training_loop_rejects_bad_train_steps(tmp_path):
    out, targets = _build_targets(tmp_path)
    with pytest.raises(ValueError, match="train_steps must be >= 1"):
        run_scalar_affine_training_loop(out, targets, k=1, train_steps=0)


def test_scalar_affine_training_loop_rejects_k_mismatch(tmp_path):
    out, targets = _build_targets(tmp_path)
    with pytest.raises(ValueError, match="requested k=2"):
        run_scalar_affine_training_loop(out, targets, k=2)


def test_training_loop_history_links_parameter_states(tmp_path):
    out, targets = _build_targets(tmp_path)
    report = run_scalar_affine_training_loop(out, targets, k=1, train_steps=5, lr=0.01)
    history = report["history"]
    for idx in range(len(history) - 1):
        assert history[idx]["alpha_after"] == history[idx + 1]["alpha_before"]
        assert history[idx]["beta_after"] == history[idx + 1]["beta_before"]
