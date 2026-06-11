from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from src.qwen_graph_prior_eval import build_matched_random_adjacency, build_qwen_topk_adjacency, run_graph_prior_eval
from src.qwen_graph_prior_quality import (
    evaluate_prior_recovery,
    gold_edges_from_block_specs,
    recovery_metrics,
    run_prior_recovery_quality,
)
from src.qwen_weight_graph import QwenWeightGraphCompiler, build_tensor_manifest_from_directory, read_weight_graph_artifacts


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


def _implanted_tensors() -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    rng = np.random.RandomState(123)
    tensor_name = "model.layers.0.self_attn.q_proj.weight"
    W = (0.01 * rng.randn(8, 8)).astype(np.float32)
    W[0:4, 4:8] = 10.0
    W[4:8, 0:4] = 9.0
    tensors = {
        tensor_name: W,
        "model.layers.0.self_attn.k_proj.weight": (0.01 * rng.randn(8, 8)).astype(np.float32),
        "model.layers.0.input_layernorm.weight": rng.randn(8).astype(np.float32),
        "model.norm.weight": rng.randn(8).astype(np.float32),
    }
    gold = [
        {
            "source_tensor": tensor_name,
            "block_in": 1,
            "block_out": 0,
            "relation": "qk_affinity_prior",
        },
        {
            "source_tensor": tensor_name,
            "block_in": 0,
            "block_out": 1,
            "relation": "qk_affinity_prior",
        },
    ]
    return tensors, gold


def _loader(tensors: dict[str, np.ndarray]):
    def load(spec):
        arr = tensors.get(spec.name)
        return arr.astype(np.float32) if arr is not None else None

    return load


def _compile_implanted_fixture(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    tensors, gold = _implanted_tensors()
    ckpt = tmp_path / "implanted_ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(_make_safetensors_bytes(tensors))
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "implanted_prior")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=1, source_model="implanted_prior")
    result = compiler.compile(specs, _loader(tensors), config_hash=config_hash, index_hash=index_hash)
    out = tmp_path / "g0"
    from src.qwen_weight_graph import write_weight_graph_artifacts

    write_weight_graph_artifacts(result, out)
    return out, gold


def test_implanted_gold_edges_resolve_to_compiled_block_edges(tmp_path):
    artifact_dir, gold_specs = _compile_implanted_fixture(tmp_path)
    result = read_weight_graph_artifacts(artifact_dir)
    gold_edges = gold_edges_from_block_specs(result, gold_specs)
    assert len(gold_edges) == 2

    qwen = build_qwen_topk_adjacency(result, k=1)
    metrics = recovery_metrics(qwen, gold_edges)
    assert metrics.recall_at_k == pytest.approx(1.0)
    assert metrics.hit_count == 2


def test_qwen_implanted_recovery_beats_matched_random(tmp_path):
    artifact_dir, gold_specs = _compile_implanted_fixture(tmp_path)
    result = read_weight_graph_artifacts(artifact_dir)
    qwen = build_qwen_topk_adjacency(result, k=1)
    random_adjacencies = [build_matched_random_adjacency(qwen, result, seed=seed) for seed in range(20)]
    gold_edges = gold_edges_from_block_specs(result, gold_specs)

    report = evaluate_prior_recovery(
        qwen_adjacency=qwen,
        random_adjacencies=random_adjacencies,
        gold_edges=gold_edges,
    )
    assert report["qwen_recall_at_k"] == pytest.approx(1.0)
    assert report["qwen_recall_at_k"] > report["random_recall_mean"]
    assert report["delta_recovery"] > 0.0
    assert report["quality_ok"] is True


def test_run_prior_recovery_quality_writes_required_report(tmp_path):
    artifact_dir, gold_specs = _compile_implanted_fixture(tmp_path)
    out = tmp_path / "graph_prior_quality_report.json"
    report = run_prior_recovery_quality(
        source_weight_graph_dir=artifact_dir,
        gold_block_specs=gold_specs,
        output_path=out,
        k=1,
        random_seeds=list(range(10)),
    )
    assert out.exists()
    written = json.loads(out.read_text())
    for key in (
        "qwen_recall_at_k",
        "qwen_precision_at_k",
        "random_recall_mean",
        "random_recall_std",
        "delta_recovery",
        "quality_ok",
    ):
        assert key in written
    assert written == report
    assert written["quality_ok"] is True


def test_graph_prior_eval_embeds_recovery_quality_in_matrix(tmp_path):
    artifact_dir, gold_specs = _compile_implanted_fixture(tmp_path)
    out = tmp_path / "prior_eval"
    result = run_graph_prior_eval(
        source_weight_graph_dir=artifact_dir,
        output_dir=out,
        k_values=[1],
        random_seeds=list(range(10)),
        gold_block_specs=gold_specs,
    )

    quality_path = out / "graph_prior_quality_report.json"
    assert quality_path.exists()
    quality_report = json.loads(quality_path.read_text())
    assert quality_report["quality_ok"] is True
    assert quality_report["delta_recovery"] > 0.0
    assert result["graph_prior_quality_report"] == quality_report
    assert result["quality_report"]["status"] == "graph_prior_quality_run"
    assert result["quality_report"]["graph_prior_quality_ok"] is True

    matrix = json.loads((out / "baseline_matrix.json").read_text())
    qwen_row = next(row for row in matrix if row["adjacency_name"] == "qwen_topk_k1")
    random_rows = [row for row in matrix if row["adjacency_name"].startswith("random_matched_k1_seed")]

    assert qwen_row["metrics_available"] is True
    assert qwen_row["quality_ok"] is True
    assert qwen_row["graph_prior_recall_at_k"] == pytest.approx(1.0)
    assert qwen_row["graph_prior_delta_recovery"] > 0.0
    assert qwen_row["graph_prior_quality_ok"] is True
    assert random_rows
    assert all(row["metrics_available"] is True for row in random_rows)
    assert all(row["graph_prior_quality_ok"] is False for row in random_rows)
    assert all("graph_prior_recall_at_k" in row for row in random_rows)


def test_missing_gold_block_spec_fails_closed(tmp_path):
    artifact_dir, _ = _compile_implanted_fixture(tmp_path)
    result = read_weight_graph_artifacts(artifact_dir)
    with pytest.raises(ValueError, match="gold block specs not found"):
        gold_edges_from_block_specs(
            result,
            [
                {
                    "source_tensor": "model.layers.0.self_attn.q_proj.weight",
                    "block_in": 99,
                    "block_out": 0,
                    "relation": "qk_affinity_prior",
                }
            ],
        )
