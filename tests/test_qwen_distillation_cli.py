from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.qwen_distillation_cli import (
    FINAL_SUMMARY_FILENAME,
    MEASURED_PROMOTION_DECISION_FILENAME,
    main,
    validate_final_summary,
)
from src.qwen_distillation_measured_gates import DEFAULT_MEASURED_GATE_REPORT_FILENAME
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
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "distillation_cli_test")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=2, source_model="distillation_cli_test")
    result = compiler.compile(specs, _loader(tensors), config_hash=config_hash, index_hash=index_hash)
    out = tmp_path / "g0"
    write_weight_graph_artifacts(result, out)
    return out


def _common_cli_args(output_root: Path) -> list[str]:
    return [
        "--output-root",
        str(output_root),
        "--k",
        "1",
        "--k-values",
        "1,2",
        "--random-seeds",
        "0,1",
        "--vocab-size",
        "16",
        "--target-seeds",
        "0,1,2",
        "--feature-dim",
        "8",
        "--forward-steps",
        "1",
        "--train-steps",
        "5",
        "--lr",
        "0.1",
        "--runtime-repeats",
        "1",
        "--max-runtime-seconds",
        "10.0",
        "--max-peak-memory-bytes",
        str(128 * 1024 * 1024),
    ]


def test_distillation_cli_runs_full_measured_pipeline(tmp_path):
    g0 = _compile_g0(tmp_path)
    output_root = tmp_path / "cli_full"

    rc = main(["--source-weight-graph-dir", str(g0), *_common_cli_args(output_root)])

    assert rc == 0
    assert (output_root / "graph_prior_eval" / "v25_handoff_manifest.json").exists()
    assert (output_root / "distillation_harness_report.json").exists()
    assert (output_root / DEFAULT_MEASURED_GATE_REPORT_FILENAME).exists()
    assert (output_root / MEASURED_PROMOTION_DECISION_FILENAME).exists()
    summary_path = output_root / FINAL_SUMMARY_FILENAME
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert validate_final_summary(summary)["promote"] is True
    assert summary["graph_prior_eval_ran"] is True
    assert summary["quality_ok"] is True
    assert summary["runtime_ok"] is True
    assert summary["memory_ok"] is True
    assert summary["safety_ok"] is True
    assert summary["promote"] is True
    assert summary["decision"] == "promoted"
    assert summary["missing_or_failed_gates"] == []


def test_distillation_cli_can_skip_graph_prior_eval_with_existing_eval_output(tmp_path):
    g0 = _compile_g0(tmp_path)
    eval_output = tmp_path / "prior_eval"
    run_graph_prior_eval(
        source_weight_graph_dir=g0,
        output_dir=eval_output,
        k_values=[1, 2],
        random_seeds=[0, 1],
        quality_mode="energy_capture",
    )
    output_root = tmp_path / "cli_existing_eval"

    rc = main(["--eval-output-dir", str(eval_output), *_common_cli_args(output_root)])

    assert rc == 0
    assert not (output_root / "graph_prior_eval").exists()
    assert (output_root / DEFAULT_MEASURED_GATE_REPORT_FILENAME).exists()
    assert (output_root / MEASURED_PROMOTION_DECISION_FILENAME).exists()
    summary = json.loads((output_root / FINAL_SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["graph_prior_eval_ran"] is False
    assert summary["source_weight_graph_dir"] is None
    assert summary["eval_output_dir"] == str(eval_output)
    assert summary["promote"] is True


def test_distillation_cli_runs_with_torch_cpu_device(tmp_path):
    g0 = _compile_g0(tmp_path)
    output_root = tmp_path / "cli_torch_cpu"

    rc = main([
        "--source-weight-graph-dir",
        str(g0),
        *_common_cli_args(output_root),
        "--device",
        "torch_cpu",
        "--max-cuda-peak-memory-bytes",
        str(128 * 1024 * 1024),
    ])

    assert rc == 0
    summary = json.loads((output_root / FINAL_SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["device"] == "torch_cpu"
    assert summary["resolved_device"] == "cpu"
    assert summary["runtime_backend"] == "torch"
    assert summary["cuda_measurement_available"] is False
    assert summary["quality_ok"] is True
    assert summary["runtime_ok"] is True
    assert summary["memory_ok"] is True
    assert summary["safety_ok"] is True
    assert summary["promote"] is True


def test_distillation_cli_rejects_bad_args(tmp_path):
    with pytest.raises(SystemExit) as missing_source:
        main(["--output-root", str(tmp_path / "missing")])
    assert missing_source.value.code == 2

    with pytest.raises(SystemExit) as bad_k:
        main(["--output-root", str(tmp_path / "bad_k"), "--source-weight-graph-dir", str(tmp_path), "--k", "0"])
    assert bad_k.value.code == 2

    with pytest.raises(SystemExit) as bad_k_values:
        main([
            "--output-root",
            str(tmp_path / "bad_k_values"),
            "--source-weight-graph-dir",
            str(tmp_path),
            "--k-values",
            "",
        ])
    assert bad_k_values.value.code == 2
