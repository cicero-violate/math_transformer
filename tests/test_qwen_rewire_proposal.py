from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.qwen_edge_trace import run_and_write_edge_trace_report
from src.qwen_graph_prior_eval import run_graph_prior_eval
from src.qwen_rewire_proposal import (
    PROPOSED_ADJACENCY_FILENAME,
    REWIRE_PROPOSAL_FILENAME,
    build_rewire_proposal_report,
    load_proposed_adjacency,
    load_rewire_proposal_report,
    main,
    run_and_write_rewire_proposal_report,
    validate_rewire_proposal_report,
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
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "rewire_proposal_test")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=2, source_model="rewire_proposal_test")
    result = compiler.compile(specs, _loader(tensors), config_hash=config_hash, index_hash=index_hash)
    out = tmp_path / "g0"
    write_weight_graph_artifacts(result, out)
    return out


def _build_eval_and_trace(tmp_path: Path) -> tuple[Path, Path]:
    g0 = _compile_g0(tmp_path)
    eval_out = tmp_path / "prior_eval"
    run_graph_prior_eval(
        source_weight_graph_dir=g0,
        output_dir=eval_out,
        k_values=[1, 2],
        random_seeds=[0, 1],
        quality_mode="energy_capture",
    )
    trace_out = tmp_path / "edge_trace"
    run_and_write_edge_trace_report(
        eval_out,
        trace_out,
        k=1,
        feature_dim=8,
        steps=1,
        seeds=[0, 1, 2],
        device="cpu",
    )
    return eval_out, trace_out


def test_rewire_proposal_builds_bounded_same_source_swaps(tmp_path):
    eval_out, trace_out = _build_eval_and_trace(tmp_path)
    report = build_rewire_proposal_report(
        eval_out,
        trace_out,
        k=1,
        max_swaps=3,
    )
    assert report["status"] == "bounded_rewire_proposal_ok"
    assert report["teacher_checkpoint_loaded"] is False
    assert report["teacher_inference_runtime_required"] is False
    assert report["raw_weight_payload_in_graph"] is False
    assert report["bounded_active_adjacency"] is True
    assert report["topology_mutated"] is False
    assert report["accepted"] is False
    assert report["proposal_bounded"] is True
    assert report["proposed_edge_count"] <= report["base_edge_count"]
    assert report["proposed_max_out_degree"] <= report["k"]
    assert report["swap_count"] <= 3
    assert validate_rewire_proposal_report(report)["proposal_bounded"] is True
    for swap in report["swaps"]:
        assert swap["src_id"] == swap["add_edge"]["src_id"]
        assert swap["drop_edge_id"] != swap["add_edge_id"]
        assert swap["drop_utility_score"] >= 0.0


def test_rewire_proposal_writes_report_and_proposed_adjacency(tmp_path):
    eval_out, trace_out = _build_eval_and_trace(tmp_path)
    out = tmp_path / "proposal"
    report = run_and_write_rewire_proposal_report(
        eval_out,
        trace_out,
        out,
        k=1,
        max_swaps=2,
    )
    assert (out / REWIRE_PROPOSAL_FILENAME).exists()
    assert (out / PROPOSED_ADJACENCY_FILENAME).exists()
    loaded = load_rewire_proposal_report(out)
    proposed = load_proposed_adjacency(out)
    assert loaded == report
    assert "proposed_adjacency" not in loaded
    assert proposed["topology_mutated"] is False
    assert proposed["accepted"] is False
    assert proposed["edge_count"] == loaded["proposed_edge_count"]
    assert proposed["max_out_degree"] <= loaded["k"]


def test_rewire_proposal_cli_writes_artifacts(tmp_path):
    eval_out, trace_out = _build_eval_and_trace(tmp_path)
    out = tmp_path / "proposal_cli"
    rc = main([
        "--eval-output-dir",
        str(eval_out),
        "--edge-trace-dir",
        str(trace_out),
        "--output-dir",
        str(out),
        "--k",
        "1",
        "--max-swaps",
        "2",
    ])
    assert rc == 0
    loaded = load_rewire_proposal_report(out)
    assert loaded["proposal_bounded"] is True
    assert loaded["accepted"] is False
    assert loaded["topology_mutated"] is False


def test_rewire_proposal_is_deterministic(tmp_path):
    eval_out, trace_out = _build_eval_and_trace(tmp_path)
    first = build_rewire_proposal_report(eval_out, trace_out, k=1, max_swaps=2)
    second = build_rewire_proposal_report(eval_out, trace_out, k=1, max_swaps=2)
    assert first["swaps"] == second["swaps"]
    assert first["proposed_adjacency"] == second["proposed_adjacency"]


def test_rewire_proposal_rejects_bad_args(tmp_path):
    eval_out, trace_out = _build_eval_and_trace(tmp_path)
    with pytest.raises(ValueError, match="max_swaps"):
        build_rewire_proposal_report(eval_out, trace_out, k=1, max_swaps=0)
    with pytest.raises(SystemExit) as bad_cli:
        main([
            "--eval-output-dir",
            str(eval_out),
            "--edge-trace-dir",
            str(trace_out),
            "--output-dir",
            str(tmp_path / "bad"),
            "--max-swaps",
            "0",
        ])
    assert bad_cli.value.code == 2
