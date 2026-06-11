from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

import numpy as np

from src.qwen_graph_prior_eval import (
    build_matched_random_adjacency,
    build_qwen_topk_adjacency,
    run_graph_prior_eval,
)
from src.qwen_weight_graph import (
    QwenWeightGraphCompiler,
    TensorSpec,
    build_tensor_manifest_from_directory,
    read_weight_graph_artifacts,
)


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


def _dense_tensors() -> dict[str, np.ndarray]:
    rng = np.random.RandomState(7)
    d = {}
    for li in range(2):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            d[f"model.layers.{li}.self_attn.{proj}.weight"] = rng.randn(8, 8).astype(np.float32)
        for proj in ("gate_proj", "up_proj"):
            d[f"model.layers.{li}.mlp.{proj}.weight"] = rng.randn(16, 8).astype(np.float32)
        d[f"model.layers.{li}.mlp.down_proj.weight"] = rng.randn(8, 16).astype(np.float32)
    d["model.embed_tokens.weight"] = rng.randn(32, 8).astype(np.float32)
    d["lm_head.weight"] = rng.randn(32, 8).astype(np.float32)
    d["model.norm.weight"] = rng.randn(8).astype(np.float32)
    return d


def _loader(tensors: dict[str, np.ndarray]):
    def load(spec: TensorSpec) -> np.ndarray | None:
        arr = tensors.get(spec.name)
        return arr.astype(np.float32) if arr is not None else None

    return load


def _compile_weight_graph(tmp_path: Path) -> Path:
    tensors = _dense_tensors()
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(_make_safetensors_bytes(tensors))
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "prior_test")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=2, source_model="prior_test")
    result = compiler.compile(specs, _loader(tensors), config_hash=config_hash, index_hash=index_hash)
    out = tmp_path / "g0"
    compiler.compile_from_directory(ckpt, out)
    # compile_from_directory exercises the same artifact path the CLI consumes.
    assert (out / "manifest.json").exists()
    assert result.manifest.raw_weight_payload_in_graph is False
    return out


def _kind_map(result):
    return {node.node_id: node.node_type for node in result.nodes}


def _coarse_distribution(adjacency, kinds):
    return sorted(
        (edge.relation, kinds[edge.src_id], kinds[edge.dst_id])
        for edge in adjacency.edges
    )


def test_qwen_topk_adjacency_is_deterministic_and_per_source_limited(tmp_path):
    artifact_dir = _compile_weight_graph(tmp_path)
    result = read_weight_graph_artifacts(artifact_dir)
    a1 = build_qwen_topk_adjacency(result, k=1)
    a2 = build_qwen_topk_adjacency(result, k=1)
    assert [edge.edge_id for edge in a1.edges] == [edge.edge_id for edge in a2.edges]

    per_source: dict[str, int] = {}
    for edge in a1.edges:
        per_source[edge.src_id] = per_source.get(edge.src_id, 0) + 1
        assert edge.source == "G_0"
        assert edge.score_name == "normalized_frobenius"
        assert "source_tensor" in edge.metadata
    assert max(per_source.values()) <= 1


def test_matched_random_preserves_budget_and_coarse_distribution(tmp_path):
    artifact_dir = _compile_weight_graph(tmp_path)
    result = read_weight_graph_artifacts(artifact_dir)
    qwen = build_qwen_topk_adjacency(result, k=2)
    random_a = build_matched_random_adjacency(qwen, result, seed=123)
    random_b = build_matched_random_adjacency(qwen, result, seed=123)
    kinds = _kind_map(result)

    assert random_a.edge_count == qwen.edge_count
    assert _coarse_distribution(random_a, kinds) == _coarse_distribution(qwen, kinds)
    assert [edge.edge_id for edge in random_a.edges] == [edge.edge_id for edge in random_b.edges]
    assert all(edge.source == "random_matched_seed_123" for edge in random_a.edges)


def test_graph_prior_eval_writes_v24_artifact_contract(tmp_path):
    artifact_dir = _compile_weight_graph(tmp_path)
    out = tmp_path / "prior"
    result = run_graph_prior_eval(
        source_weight_graph_dir=artifact_dir,
        output_dir=out,
        k_values=[1, 2],
        random_seeds=[0, 1],
        quality_dataset="data/synthetic_hard/val.jsonl",
    )

    expected = {
        "prior_config.json",
        "baseline_matrix.json",
        "baseline_matrix.csv",
        "adjacency_summary.json",
        "quality_report.json",
        "memory_report.json",
        "runtime_report.json",
        "paired_regression_report.jsonl",
        "promotion_decision.json",
    }
    assert expected.issubset({p.name for p in out.iterdir()})
    assert result["prior_config"]["teacher_checkpoint_loaded"] is False
    assert result["promotion_decision"]["promote"] is False
    assert result["promotion_decision"]["quality_ok"] is False

    matrix = json.loads((out / "baseline_matrix.json").read_text())
    names = {row["adjacency_name"] for row in matrix}
    assert {"dense_full", "hand_topology_k4", "learned_topology_k4", "qwen_topk_k1", "qwen_topk_k2"}.issubset(names)
    assert any(name.startswith("random_matched_k1_seed") for name in names)
    assert all(row["quality_ok"] is False for row in matrix)
    assert all(row["speed_ok"] is False for row in matrix)

    with (out / "baseline_matrix.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(matrix)


def test_graph_scope_filters_moe_nodes_out_of_attention_mlp(tmp_path):
    artifact_dir = _compile_weight_graph(tmp_path)
    result = read_weight_graph_artifacts(artifact_dir)
    adj = build_qwen_topk_adjacency(result, k=2, graph_scope="attention_mlp")
    kinds = _kind_map(result)
    assert all(kinds[edge.src_id] not in {"expert", "router"} for edge in adj.edges)
    assert all(kinds[edge.dst_id] not in {"expert", "router"} for edge in adj.edges)
