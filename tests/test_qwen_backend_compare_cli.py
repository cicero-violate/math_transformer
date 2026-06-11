from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.qwen_backend_compare_cli import (
    BACKEND_COMPARE_REPORT_FILENAME,
    build_backend_compare_report,
    main,
    validate_backend_compare_report,
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
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "backend_compare_test")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=2, source_model="backend_compare_test")
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


def test_backend_compare_report_checks_python_torch_auto_parity(tmp_path):
    eval_out = _build_eval_output(tmp_path)
    report = build_backend_compare_report(
        eval_out,
        k=1,
        feature_dim=8,
        steps=1,
        seeds=[0, 1, 2],
        devices=["cpu", "torch_cpu", "auto"],
        runtime_repeats=1,
    )
    assert report["status"] == "backend_compare_ok"
    assert report["baseline_device"] == "cpu"
    assert report["available_device_count"] == 3
    assert report["parity_ok"] is True
    assert report["runtime_ok"] is True
    assert report["memory_ok"] is True
    assert validate_backend_compare_report(report)["parity_ok"] is True
    by_device = {entry["device"]: entry for entry in report["device_reports"]}
    assert by_device["cpu"]["device_info"]["runtime_backend"] == "python"
    assert by_device["torch_cpu"]["device_info"]["runtime_backend"] == "torch"
    assert by_device["torch_cpu"]["max_abs_diff"] == 0.0
    assert by_device["auto"]["available"] is True


def test_backend_compare_cli_writes_report(tmp_path):
    eval_out = _build_eval_output(tmp_path)
    output_dir = tmp_path / "compare"
    rc = main([
        "--eval-output-dir",
        str(eval_out),
        "--output-dir",
        str(output_dir),
        "--k",
        "1",
        "--feature-dim",
        "8",
        "--steps",
        "1",
        "--seeds",
        "0,1",
        "--devices",
        "cpu,torch_cpu,auto",
        "--runtime-repeats",
        "1",
    ])
    assert rc == 0
    path = output_dir / BACKEND_COMPARE_REPORT_FILENAME
    assert path.exists()
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["parity_ok"] is True
    assert report["available_device_count"] == 3


def test_backend_compare_records_unavailable_cuda_without_failing_when_cpu_available(tmp_path):
    eval_out = _build_eval_output(tmp_path)
    report = build_backend_compare_report(
        eval_out,
        k=1,
        feature_dim=8,
        steps=1,
        seeds=[0],
        devices=["cpu", "cuda"],
        runtime_repeats=1,
    )
    by_device = {entry["device"]: entry for entry in report["device_reports"]}
    assert by_device["cpu"]["available"] is True
    if by_device["cuda"]["available"]:
        assert by_device["cuda"]["device_info"]["resolved_device"] == "cuda"
    else:
        assert by_device["cuda"]["status"] == "backend_unavailable"
        assert "cuda device requested" in by_device["cuda"]["reason"]
    assert report["available_device_count"] >= 1


def test_backend_compare_rejects_bad_args(tmp_path):
    eval_out = _build_eval_output(tmp_path)
    with pytest.raises(ValueError, match="runtime_repeats"):
        build_backend_compare_report(eval_out, runtime_repeats=0)
    with pytest.raises(ValueError, match="parity_tolerance"):
        build_backend_compare_report(eval_out, parity_tolerance=-1.0)
    with pytest.raises(SystemExit) as bad_devices:
        main(["--eval-output-dir", str(eval_out), "--output-dir", str(tmp_path / "bad"), "--devices", "cpu,bogus"])
    assert bad_devices.value.code == 2
