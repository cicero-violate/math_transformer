from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.qwen_distillation_targets import (
    iter_frozen_target_rows,
    validate_frozen_distillation_targets,
    write_frozen_distillation_targets,
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
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "distillation_targets_test")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=2, source_model="distillation_targets_test")
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


def _read_manifest(target_out: Path) -> dict:
    return json.loads((target_out / "frozen_targets_manifest.json").read_text(encoding="utf-8"))


def _write_manifest(target_out: Path, manifest: dict) -> None:
    (target_out / "frozen_targets_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rewrite_rows_and_refresh_manifest(target_out: Path, rows: list[dict]) -> None:
    rows_path = target_out / "frozen_targets.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = _read_manifest(target_out)
    manifest["target_rows_sha256"] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    _write_manifest(target_out, manifest)


def test_write_and_validate_frozen_distillation_targets(tmp_path):
    out = _build_eval_output(tmp_path)
    target_out = tmp_path / "targets"
    manifest = write_frozen_distillation_targets(out, target_out, k=1, feature_dim=8, seeds=[0, 1, 2])
    assert (target_out / "frozen_targets_manifest.json").exists()
    assert (target_out / "frozen_targets.jsonl").exists()
    assert manifest["status"] == "frozen_targets_ready"
    assert manifest["teacher_checkpoint_loaded_at_runtime"] is False
    assert manifest["teacher_inference_runtime_required"] is False
    assert manifest["raw_weight_payload_in_graph"] is False
    assert manifest["student_training_started"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["feature_dim"] == 8
    assert manifest["row_count"] == 3
    assert manifest["selected_adjacency_name"] == "qwen_topk_k1"
    assert manifest["selected_adjacency_k"] == 1
    summary = validate_frozen_distillation_targets(target_out)
    assert summary["status"] == "frozen_targets_valid"


def test_frozen_targets_are_deterministic(tmp_path):
    out = _build_eval_output(tmp_path)
    first_dir = tmp_path / "targets_a"
    second_dir = tmp_path / "targets_b"
    first = write_frozen_distillation_targets(out, first_dir, k=1, feature_dim=8, seeds=[0, 1, 2])
    second = write_frozen_distillation_targets(out, second_dir, k=1, feature_dim=8, seeds=[0, 1, 2])
    assert first["target_rows_sha256"] == second["target_rows_sha256"]
    first_checksums = [row["target_checksum"] for row in iter_frozen_target_rows(first_dir)]
    second_checksums = [row["target_checksum"] for row in iter_frozen_target_rows(second_dir)]
    assert first_checksums == second_checksums


def test_frozen_targets_change_with_seed(tmp_path):
    out = _build_eval_output(tmp_path)
    first = write_frozen_distillation_targets(out, tmp_path / "targets_a", k=1, feature_dim=8, seeds=[0])
    second = write_frozen_distillation_targets(out, tmp_path / "targets_b", k=1, feature_dim=8, seeds=[1])
    assert first["target_rows_sha256"] != second["target_rows_sha256"]


def test_iter_frozen_target_rows_reads_rows(tmp_path):
    out = _build_eval_output(tmp_path)
    target_out = tmp_path / "targets"
    write_frozen_distillation_targets(out, target_out, k=1, feature_dim=8, seeds=[0, 1])
    rows = list(iter_frozen_target_rows(target_out))
    assert len(rows) == 2
    assert all("target_features" in row for row in rows)
    assert all("target_checksum" in row for row in rows)


def test_validate_rejects_manifest_runtime_teacher_true(tmp_path):
    out = _build_eval_output(tmp_path)
    target_out = tmp_path / "targets"
    write_frozen_distillation_targets(out, target_out, k=1, feature_dim=8, seeds=[0, 1])
    manifest = _read_manifest(target_out)
    manifest["teacher_checkpoint_loaded_at_runtime"] = True
    _write_manifest(target_out, manifest)
    with pytest.raises(ValueError, match="teacher_checkpoint_loaded_at_runtime"):
        validate_frozen_distillation_targets(target_out)


def test_validate_rejects_bad_row_checksum(tmp_path):
    out = _build_eval_output(tmp_path)
    target_out = tmp_path / "targets"
    write_frozen_distillation_targets(out, target_out, k=1, feature_dim=8, seeds=[0, 1])
    rows = list(iter_frozen_target_rows(target_out))
    node_id = sorted(rows[0]["target_features"])[0]
    rows[0]["target_features"][node_id][0] += 1.0
    _rewrite_rows_and_refresh_manifest(target_out, rows)
    with pytest.raises(ValueError, match="target_checksum mismatch"):
        validate_frozen_distillation_targets(target_out)


def test_validate_rejects_forbidden_raw_payload_key(tmp_path):
    out = _build_eval_output(tmp_path)
    target_out = tmp_path / "targets"
    write_frozen_distillation_targets(out, target_out, k=1, feature_dim=8, seeds=[0, 1])
    rows = list(iter_frozen_target_rows(target_out))
    rows[0]["metadata"]["raw_weight_payload"] = [1, 2, 3]
    _rewrite_rows_and_refresh_manifest(target_out, rows)
    with pytest.raises(ValueError, match="raw tensor payload field forbidden"):
        validate_frozen_distillation_targets(target_out)
