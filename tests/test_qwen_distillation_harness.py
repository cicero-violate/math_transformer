from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.qwen_distillation_harness import (
    load_distillation_harness_report,
    run_fixed_topology_distillation_harness,
    validate_distillation_harness_report,
)
from src.qwen_graph_prior_eval import run_graph_prior_eval
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
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "distillation_harness_test")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=2, source_model="distillation_harness_test")
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


def _core_report(report: dict) -> dict:
    ignored = {"eval_output_dir", "output_dir", "logit_targets_dir", "artifacts"}
    return {key: value for key, value in report.items() if key not in ignored}


def test_distillation_harness_runs_end_to_end(tmp_path):
    out = _build_eval_output(tmp_path)
    harness_out = tmp_path / "harness"
    report = run_fixed_topology_distillation_harness(
        out,
        harness_out,
        k=1,
        vocab_size=16,
        target_seeds=[0, 1, 2],
        train_steps=5,
        lr=0.1,
    )
    assert report["status"] == "fixed_topology_distillation_harness_ok"
    assert report["student_training_started"] is True
    assert report["teacher_checkpoint_loaded"] is False
    assert report["teacher_inference_runtime_required"] is False
    assert report["teacher_distillation_started"] is False
    assert report["kl_training_started"] is True
    assert report["raw_weight_payload_in_graph"] is False
    assert report["bounded_active_adjacency"] is True
    assert report["promotion_eligible"] is False
    assert report["promotion_decision"] == "not_promoted"
    assert report["k"] == 1
    assert report["vocab_size"] == 16
    assert report["target_row_count"] == 3
    assert report["kl_before"] >= 0.0
    assert report["kl_after"] >= 0.0
    assert report["kl_after"] <= report["kl_before"]
    assert report["kl_decreased"] is True
    assert report["monotonic_nonincreasing"] is True
    assert report["finite"] is True
    for rel_path in report["artifacts"].values():
        assert (harness_out / rel_path).exists()


def test_distillation_harness_is_deterministic(tmp_path):
    out = _build_eval_output(tmp_path)
    first = run_fixed_topology_distillation_harness(out, tmp_path / "harness_a", k=1, target_seeds=[0, 1, 2])
    second = run_fixed_topology_distillation_harness(out, tmp_path / "harness_b", k=1, target_seeds=[0, 1, 2])
    assert _core_report(first) == _core_report(second)


def test_distillation_harness_uses_existing_logit_targets_without_mutation(tmp_path):
    out = _build_eval_output(tmp_path)
    external = tmp_path / "external_logit_targets"
    manifest = write_frozen_logit_distillation_targets(external, vocab_size=16, seeds=[0, 1, 2], temperature=1.0)
    manifest_bytes = (external / "frozen_logit_targets_manifest.json").read_bytes()
    rows_bytes = (external / "frozen_logit_targets.jsonl").read_bytes()
    report = run_fixed_topology_distillation_harness(
        out,
        tmp_path / "harness",
        k=1,
        logit_targets_dir=external,
    )
    assert report["used_existing_logit_targets"] is True
    assert manifest["target_rows_sha256"] == json.loads(manifest_bytes.decode("utf-8"))["target_rows_sha256"]
    assert (external / "frozen_logit_targets_manifest.json").read_bytes() == manifest_bytes
    assert (external / "frozen_logit_targets.jsonl").read_bytes() == rows_bytes


def test_distillation_harness_writes_and_loads_report(tmp_path):
    out = _build_eval_output(tmp_path)
    harness_out = tmp_path / "harness"
    report = run_fixed_topology_distillation_harness(out, harness_out, k=1, target_seeds=[0, 1, 2])
    loaded = load_distillation_harness_report(harness_out / "distillation_harness_report.json")
    assert loaded == report
    summary = validate_distillation_harness_report(loaded)
    assert summary["status"] == "fixed_topology_distillation_harness_valid"


def test_distillation_harness_rejects_bad_train_steps_lr_temperature(tmp_path):
    out = _build_eval_output(tmp_path)
    with pytest.raises(ValueError, match="train_steps must be >= 1"):
        run_fixed_topology_distillation_harness(out, tmp_path / "bad_steps", k=1, train_steps=0)
    with pytest.raises(ValueError, match="lr must be > 0"):
        run_fixed_topology_distillation_harness(out, tmp_path / "bad_lr", k=1, lr=0.0)
    with pytest.raises(ValueError, match="temperature must be > 0"):
        run_fixed_topology_distillation_harness(out, tmp_path / "bad_temp", k=1, temperature=0.0)


def test_distillation_harness_rejects_missing_handoff(tmp_path):
    with pytest.raises(ValueError, match="handoff"):
        run_fixed_topology_distillation_harness(tmp_path / "missing_eval", tmp_path / "harness", k=1)
