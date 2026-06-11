from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.qwen_edge_trace import (
    EDGE_TRACE_REPORT_FILENAME,
    EDGE_TRACE_ROWS_FILENAME,
    EDGE_UTILITY_SUMMARY_FILENAME,
    build_edge_trace_report,
    load_edge_trace_report,
    load_edge_trace_rows,
    main,
    run_and_write_edge_trace_report,
    validate_edge_trace_report,
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
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "edge_trace_test")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=2, source_model="edge_trace_test")
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


def test_edge_trace_report_collects_one_row_per_edge_seed_step(tmp_path):
    eval_out = _build_eval_output(tmp_path)
    report = build_edge_trace_report(
        eval_out,
        k=1,
        feature_dim=8,
        steps=2,
        seeds=[0, 1, 2],
        device="torch_cpu",
    )
    assert report["status"] == "edge_trace_ok"
    assert report["teacher_checkpoint_loaded"] is False
    assert report["teacher_inference_runtime_required"] is False
    assert report["raw_weight_payload_in_graph"] is False
    assert report["bounded_active_adjacency"] is True
    assert report["topology_mutated"] is False
    assert report["row_count"] == report["edge_count"] * 3 * 2
    assert len(report["edge_trace_rows"]) == report["row_count"]
    assert report["device_info"]["runtime_backend"] == "torch"
    assert report["trace_backend"] == "python_utility_probe"
    assert report["edge_utility_summary"]["edge_count"] == report["edge_count"]
    assert report["edge_utility_summary"]["finite"] is True
    assert validate_edge_trace_report(report)["row_count"] == report["row_count"]
    first_row = report["edge_trace_rows"][0]
    assert first_row["used"] is True
    assert first_row["message_l1"] >= 0.0
    assert first_row["dst_delta_l1"] == first_row["message_l1"]


def test_edge_trace_writes_report_rows_and_summary(tmp_path):
    eval_out = _build_eval_output(tmp_path)
    out = tmp_path / "edge_trace"
    report = run_and_write_edge_trace_report(
        eval_out,
        out,
        k=1,
        feature_dim=8,
        steps=1,
        seeds=[0, 1],
        device="cpu",
    )
    assert (out / EDGE_TRACE_REPORT_FILENAME).exists()
    assert (out / EDGE_TRACE_ROWS_FILENAME).exists()
    assert (out / EDGE_UTILITY_SUMMARY_FILENAME).exists()
    loaded = load_edge_trace_report(out)
    rows = load_edge_trace_rows(out)
    assert "edge_trace_rows" not in loaded
    assert loaded == report
    assert len(rows) == loaded["row_count"]
    assert json.loads((out / EDGE_UTILITY_SUMMARY_FILENAME).read_text(encoding="utf-8"))["row_count"] == loaded["row_count"]
    assert validate_edge_trace_report(loaded)["status"] == "edge_trace_report_valid"


def test_edge_trace_cli_writes_artifacts(tmp_path):
    eval_out = _build_eval_output(tmp_path)
    out = tmp_path / "edge_trace_cli"
    rc = main([
        "--eval-output-dir",
        str(eval_out),
        "--output-dir",
        str(out),
        "--k",
        "1",
        "--feature-dim",
        "8",
        "--steps",
        "1",
        "--seeds",
        "0,1",
        "--device",
        "torch_cpu",
    ])
    assert rc == 0
    loaded = load_edge_trace_report(out)
    assert loaded["device"] == "torch_cpu"
    assert loaded["row_count"] == loaded["edge_count"] * 2
    assert (out / EDGE_TRACE_ROWS_FILENAME).exists()


def test_edge_trace_is_deterministic(tmp_path):
    eval_out = _build_eval_output(tmp_path)
    first = build_edge_trace_report(eval_out, k=1, feature_dim=8, steps=2, seeds=[0, 1])
    second = build_edge_trace_report(eval_out, k=1, feature_dim=8, steps=2, seeds=[0, 1])
    assert first["initial_input_checksum_by_seed"] == second["initial_input_checksum_by_seed"]
    assert first["final_output_checksum_by_seed"] == second["final_output_checksum_by_seed"]
    assert first["edge_trace_rows"] == second["edge_trace_rows"]
    assert first["edge_utility_summary"] == second["edge_utility_summary"]


def test_edge_trace_rejects_bad_args(tmp_path):
    eval_out = _build_eval_output(tmp_path)
    with pytest.raises(ValueError, match="feature_dim"):
        build_edge_trace_report(eval_out, k=1, feature_dim=0)
    with pytest.raises(ValueError, match="steps"):
        build_edge_trace_report(eval_out, k=1, steps=0)
    with pytest.raises(ValueError, match="seeds"):
        build_edge_trace_report(eval_out, k=1, seeds=[])
    with pytest.raises(SystemExit) as bad_cli:
        main(["--eval-output-dir", str(eval_out), "--output-dir", str(tmp_path / "bad"), "--seeds", ""])
    assert bad_cli.value.code == 2
