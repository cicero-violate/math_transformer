from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from src.qwen_graph_prior_eval import run_graph_prior_eval
from src.qwen_kl_distillation_trainable import (
    kl_loss_and_bias_grads,
    run_and_write_logit_bias_training_report,
    run_logit_bias_train_step,
    run_logit_bias_training_loop,
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
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "kl_trainable_test")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=2, source_model="kl_trainable_test")
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


def test_logit_bias_train_step_reduces_kl(tmp_path):
    out, targets = _build_eval_and_logit_targets(tmp_path)
    report = run_logit_bias_train_step(out, targets, k=1, lr=0.1)
    assert report["status"] == "logit_bias_train_step_ok"
    assert report["student_training_started"] is True
    assert report["teacher_checkpoint_loaded"] is False
    assert report["teacher_inference_runtime_required"] is False
    assert report["teacher_distillation_started"] is False
    assert report["kl_training_started"] is True
    assert report["raw_weight_payload_in_graph"] is False
    assert report["bounded_active_adjacency"] is True
    assert report["promotion_eligible"] is False
    assert report["optimizer"] == "analytic_sgd_logit_bias_kl"
    assert report["vocab_size"] == 16
    assert report["row_count"] == 3
    assert report["kl_before"] >= 0.0
    assert report["kl_after"] >= 0.0
    assert report["kl_after"] <= report["kl_before"]
    assert report["kl_decreased"] is True
    assert report["bias_before_checksum"] != report["bias_after_checksum"]
    assert report["finite"] is True


def test_logit_bias_training_loop_reduces_kl(tmp_path):
    out, targets = _build_eval_and_logit_targets(tmp_path)
    report = run_logit_bias_training_loop(out, targets, k=1, train_steps=5, lr=0.1)
    assert report["status"] == "logit_bias_training_loop_ok"
    assert report["train_steps"] == 5
    assert len(report["history"]) == 5
    assert report["kl_final"] <= report["kl_initial"]
    assert report["kl_decreased"] is True
    assert report["monotonic_nonincreasing"] is True
    assert report["finite"] is True


def test_logit_bias_training_loop_threads_device(tmp_path):
    out, targets = _build_eval_and_logit_targets(tmp_path)
    report = run_logit_bias_training_loop(out, targets, k=1, train_steps=2, lr=0.1, device="torch_cpu")
    assert report["status"] == "logit_bias_training_loop_ok"
    assert report["device"] == "torch_cpu"
    assert report["finite"] is True


def test_logit_bias_training_write_helper_threads_device(tmp_path):
    out, targets = _build_eval_and_logit_targets(tmp_path)
    report_path = tmp_path / "reports" / "kl_training_torch_cpu.json"
    report = run_and_write_logit_bias_training_report(
        out,
        targets,
        report_path,
        k=1,
        train_steps=2,
        lr=0.1,
        device="torch_cpu",
    )
    assert report_path.exists()
    assert report["device"] == "torch_cpu"


def test_logit_bias_training_loop_is_deterministic(tmp_path):
    out, targets = _build_eval_and_logit_targets(tmp_path)
    first = run_logit_bias_training_loop(out, targets, k=1, train_steps=5, lr=0.1)
    second = run_logit_bias_training_loop(out, targets, k=1, train_steps=5, lr=0.1)
    assert first["bias_final_checksum"] == second["bias_final_checksum"]
    assert first["kl_final"] == second["kl_final"]
    assert first["cross_entropy_final"] == second["cross_entropy_final"]
    assert first["history"] == second["history"]


def test_logit_bias_training_report_writes_json(tmp_path):
    out, targets = _build_eval_and_logit_targets(tmp_path)
    report_path = tmp_path / "reports" / "kl_training.json"
    report = run_and_write_logit_bias_training_report(out, targets, report_path, k=1, train_steps=5, lr=0.1)
    assert report_path.exists()
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded == report


def test_kl_loss_and_bias_grads_simple_case():
    metrics = kl_loss_and_bias_grads(
        [0.0, 0.0],
        [1.0, 0.0],
        bias=[0.0, 0.0],
        temperature=1.0,
    )
    assert metrics["kl"] == pytest.approx(math.log(2.0), abs=1e-10)
    assert metrics["grad_bias"] == pytest.approx([-0.5, 0.5])
    assert metrics["finite"] is True


def test_logit_bias_train_step_rejects_bad_lr_bias_temperature(tmp_path):
    out, targets = _build_eval_and_logit_targets(tmp_path)
    with pytest.raises(ValueError, match="lr must be > 0"):
        run_logit_bias_train_step(out, targets, k=1, lr=0.0)
    with pytest.raises(ValueError, match="lr must be finite"):
        run_logit_bias_train_step(out, targets, k=1, lr=float("nan"))
    with pytest.raises(ValueError, match="temperature must be > 0"):
        run_logit_bias_train_step(out, targets, k=1, temperature=0.0)
    with pytest.raises(ValueError, match="bias length"):
        run_logit_bias_train_step(out, targets, k=1, bias=[0.0])
    with pytest.raises(ValueError, match="bias must be finite"):
        run_logit_bias_train_step(out, targets, k=1, bias=[0.0] * 15 + [float("inf")])


def test_logit_bias_training_loop_rejects_bad_train_steps(tmp_path):
    out, targets = _build_eval_and_logit_targets(tmp_path)
    with pytest.raises(ValueError, match="train_steps must be >= 1"):
        run_logit_bias_training_loop(out, targets, k=1, train_steps=0)


def test_logit_bias_train_step_rejects_invalid_targets(tmp_path):
    out, targets = _build_eval_and_logit_targets(tmp_path)
    manifest_path = targets / "frozen_logit_targets_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["teacher_checkpoint_loaded_at_runtime"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="teacher_checkpoint_loaded_at_runtime"):
        run_logit_bias_train_step(out, targets, k=1)
