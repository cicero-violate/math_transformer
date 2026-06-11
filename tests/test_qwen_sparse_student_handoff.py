from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.qwen_graph_prior_eval import run_graph_prior_eval
from src.qwen_sparse_student_handoff import (
    build_fixed_topology_student_stub,
    list_selected_adjacencies,
    load_selected_adjacency,
    load_v25_handoff,
    validate_selected_adjacency,
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
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "handoff_test")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=2, source_model="handoff_test")
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


def _valid_selected_adjacency(k: int = 1) -> dict:
    return {
        "schema_version": "qwen_selected_adjacency.v1",
        "adjacency_name": f"qwen_topk_k{k}",
        "source": "G_0",
        "k": k,
        "edge_count": 1,
        "node_count": 2,
        "bounded": True,
        "selection_policy": "per_source_topk_score_desc",
        "edge_score_name": "normalized_frobenius",
        "graph_scope": "attention_mlp_moe",
        "edges": [
            {
                "edge_id": "e1",
                "src_id": "a",
                "dst_id": "b",
                "relation": "qk_affinity_prior",
                "weight": 1.0,
                "score_name": "normalized_frobenius",
                "source": "G_0",
                "metadata": {"source_tensor": "t", "provenance": {"block_in": 0, "block_out": 0}},
            }
        ],
    }


def test_fixed_topology_student_stub_loads_v24_handoff(tmp_path):
    out = _build_eval_output(tmp_path)
    stub = build_fixed_topology_student_stub(out, k=1)
    assert stub["status"] == "fixed_topology_stub_ok"
    assert stub["student_training_started"] is False
    assert stub["teacher_checkpoint_loaded"] is False
    assert stub["raw_weight_payload_in_graph"] is False
    assert stub["bounded_active_adjacency"] is True
    assert stub["adjacency_name"] == "qwen_topk_k1"
    assert stub["k"] == 1
    assert stub["edge_count"] > 0
    assert stub["max_out_degree"] <= 1
    assert stub["ready_for_v25_distillation"] is True


def test_handoff_loader_rejects_raw_payload_true(tmp_path):
    (tmp_path / "v25_handoff_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "qwen_v25_handoff.v1",
                "teacher_checkpoint_loaded": False,
                "raw_weight_payload_in_graph": True,
                "bounded_active_adjacency": True,
                "student_training_started": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="raw_weight_payload_in_graph"):
        load_v25_handoff(tmp_path)


def test_selected_adjacency_validator_rejects_unbounded_degree():
    adjacency = _valid_selected_adjacency(k=1)
    adjacency["edge_count"] = 2
    adjacency["node_count"] = 3
    adjacency["edges"].append(
        {
            "edge_id": "e2",
            "src_id": "a",
            "dst_id": "c",
            "relation": "qk_affinity_prior",
            "weight": 0.5,
            "score_name": "normalized_frobenius",
            "source": "G_0",
            "metadata": {},
        }
    )
    with pytest.raises(ValueError, match="exceeds k=1"):
        validate_selected_adjacency(adjacency)


def test_selected_adjacency_validator_rejects_raw_tensor_payload():
    adjacency = _valid_selected_adjacency(k=1)
    for forbidden_key in ("tensor_values", "raw_weight", "weight_payload", "payload"):
        bad = json.loads(json.dumps(adjacency))
        bad["edges"][0][forbidden_key] = [1.0, 2.0]
        with pytest.raises(ValueError, match="raw tensor payload field forbidden"):
            validate_selected_adjacency(bad)


def test_selected_adjacency_loader_selects_by_k_or_name(tmp_path):
    out = _build_eval_output(tmp_path)
    rows = list_selected_adjacencies(out)
    assert {row["adjacency_name"] for row in rows} == {"qwen_topk_k1", "qwen_topk_k2"}
    by_k = load_selected_adjacency(out, k=2)
    by_name = load_selected_adjacency(out, adjacency_name="qwen_topk_k1")
    assert by_k["adjacency_name"] == "qwen_topk_k2"
    assert by_k["k"] == 2
    assert by_name["adjacency_name"] == "qwen_topk_k1"
    assert by_name["k"] == 1


def test_fixed_topology_student_stub_reports_missing_handoff(tmp_path):
    stub = build_fixed_topology_student_stub(tmp_path)
    assert stub["status"] == "handoff_missing"
    assert stub["student_training_started"] is False
    assert stub["teacher_checkpoint_loaded"] is False
