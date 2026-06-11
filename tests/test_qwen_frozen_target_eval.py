from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.qwen_distillation_targets import checksum_json, iter_frozen_target_rows, write_frozen_distillation_targets
from src.qwen_frozen_target_eval import (
    evaluate_fixed_topology_against_frozen_targets,
    run_and_write_frozen_target_eval_report,
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
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "frozen_target_eval_test")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=2, source_model="frozen_target_eval_test")
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
    write_frozen_distillation_targets(out, targets, k=1, feature_dim=8, seeds=[0, 1, 2])
    return out, targets


def _read_manifest(targets: Path) -> dict:
    return json.loads((targets / "frozen_targets_manifest.json").read_text(encoding="utf-8"))


def _write_manifest(targets: Path, manifest: dict) -> None:
    (targets / "frozen_targets_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rewrite_rows_and_refresh_manifest(targets: Path, rows: list[dict]) -> None:
    rows_path = targets / "frozen_targets.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = _read_manifest(targets)
    manifest["target_rows_sha256"] = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    _write_manifest(targets, manifest)


def test_evaluate_fixed_topology_against_frozen_targets(tmp_path):
    out, targets = _build_targets(tmp_path)
    report = evaluate_fixed_topology_against_frozen_targets(out, targets)
    assert report["status"] == "frozen_target_eval_ok"
    assert report["student_training_started"] is False
    assert report["teacher_checkpoint_loaded"] is False
    assert report["teacher_inference_runtime_required"] is False
    assert report["teacher_distillation_started"] is False
    assert report["raw_weight_payload_in_graph"] is False
    assert report["bounded_active_adjacency"] is True
    assert report["promotion_eligible"] is False
    assert report["adjacency_name"] == "qwen_topk_k1"
    assert report["k"] == 1
    assert report["feature_dim"] == 8
    assert report["row_count"] == 3
    assert report["loss_mse_mean"] >= 0.0
    assert report["loss_l1_mean"] >= 0.0
    assert report["finite"] is True
    assert len(report["rows"]) == 3
    for row in report["rows"]:
        assert {"row_id", "seed", "loss_mse", "loss_l1", "output_checksum", "target_checksum"}.issubset(row)


def test_frozen_target_eval_is_deterministic(tmp_path):
    out, targets = _build_targets(tmp_path)
    first = evaluate_fixed_topology_against_frozen_targets(out, targets)
    second = evaluate_fixed_topology_against_frozen_targets(out, targets)
    assert first["loss_mse_mean"] == second["loss_mse_mean"]
    assert first["loss_l1_mean"] == second["loss_l1_mean"]
    assert first["rows"] == second["rows"]


def test_frozen_target_eval_writes_report(tmp_path):
    out, targets = _build_targets(tmp_path)
    report_path = tmp_path / "reports" / "frozen_target_eval.json"
    report = run_and_write_frozen_target_eval_report(out, targets, report_path)
    assert report_path.exists()
    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded == report


def test_frozen_target_eval_rejects_k_mismatch(tmp_path):
    out, targets = _build_targets(tmp_path)
    with pytest.raises(ValueError, match="requested k=2"):
        evaluate_fixed_topology_against_frozen_targets(out, targets, k=2)


def test_frozen_target_eval_uses_frozen_targets_not_regenerated_targets(tmp_path):
    out, targets = _build_targets(tmp_path)
    original = evaluate_fixed_topology_against_frozen_targets(out, targets)

    rows = list(iter_frozen_target_rows(targets))
    node_id = sorted(rows[0]["target_features"])[0]
    rows[0]["target_features"][node_id][0] += 3.0
    rows[0]["target_checksum"] = checksum_json(rows[0]["target_features"])
    _rewrite_rows_and_refresh_manifest(targets, rows)

    changed = evaluate_fixed_topology_against_frozen_targets(out, targets)
    assert changed["rows"][0]["target_checksum"] != original["rows"][0]["target_checksum"]
    assert changed["rows"][0]["loss_mse"] != original["rows"][0]["loss_mse"]


def test_frozen_target_eval_rejects_bad_steps(tmp_path):
    out, targets = _build_targets(tmp_path)
    with pytest.raises(ValueError, match="steps must be >= 1"):
        evaluate_fixed_topology_against_frozen_targets(out, targets, steps=0)
