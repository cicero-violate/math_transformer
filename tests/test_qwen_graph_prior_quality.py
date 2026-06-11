from __future__ import annotations

import json
import subprocess
import struct
import sys
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


def _compile_energy_proxy_fixture(tmp_path: Path) -> Path:
    tensors = _energy_proxy_tensors()
    ckpt = tmp_path / "energy_proxy_ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(_make_safetensors_bytes(tensors))
    specs, config_hash, index_hash = build_tensor_manifest_from_directory(ckpt, "energy_proxy")
    compiler = QwenWeightGraphCompiler(block_size=4, topk=2, source_model="energy_proxy")
    result = compiler.compile(specs, _loader(tensors), config_hash=config_hash, index_hash=index_hash)
    out = tmp_path / "g0_energy"
    from src.qwen_weight_graph import write_weight_graph_artifacts

    write_weight_graph_artifacts(result, out)
    return out


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
        quality_mode="implanted_recovery",
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


def test_graph_prior_eval_implanted_recovery_requires_gold_specs(tmp_path):
    artifact_dir, _ = _compile_implanted_fixture(tmp_path)
    with pytest.raises(ValueError, match="requires gold_block_specs"):
        run_graph_prior_eval(
            source_weight_graph_dir=artifact_dir,
            output_dir=tmp_path / "prior_eval_no_gold",
            k_values=[1],
            random_seeds=[0],
            quality_mode="implanted_recovery",
        )


def test_graph_prior_eval_cli_artifact_contract_implanted_recovery(tmp_path):
    artifact_dir, gold_specs = _compile_implanted_fixture(tmp_path)
    gold_specs_path = tmp_path / "gold_block_specs.json"
    gold_specs_path.write_text(json.dumps({"gold_edges": gold_specs}, indent=2), encoding="utf-8")
    out = tmp_path / "prior_eval_cli"
    project_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.qwen_graph_prior_eval",
            "--source-weight-graph-dir",
            str(artifact_dir),
            "--output-dir",
            str(out),
            "--k-values",
            "1",
            "--random-seeds",
            "0,1,2,3,4,5,6,7,8,9",
            "--graph-scope",
            "attention_mlp_moe",
            "--quality-mode",
            "implanted_recovery",
            "--gold-specs",
            str(gold_specs_path),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    expected_files = {
        "prior_config.json",
        "baseline_matrix.json",
        "baseline_matrix.csv",
        "adjacency_summary.json",
        "quality_report.json",
        "graph_prior_quality_report.json",
        "memory_report.json",
        "runtime_report.json",
        "paired_regression_report.jsonl",
        "promotion_decision.json",
    }
    assert expected_files.issubset({path.name for path in out.iterdir()})

    quality_report = json.loads((out / "graph_prior_quality_report.json").read_text())
    assert quality_report["quality_ok"] is True
    assert quality_report["qwen_recall_at_k"] == pytest.approx(1.0)
    assert quality_report["delta_recovery"] > 0.0

    baseline_matrix = json.loads((out / "baseline_matrix.json").read_text())
    qwen_row = next(row for row in baseline_matrix if row["adjacency_name"] == "qwen_topk_k1")
    random_rows = [
        row
        for row in baseline_matrix
        if row["adjacency_name"].startswith("random_matched_k1_seed")
    ]
    assert qwen_row["metrics_available"] is True
    assert qwen_row["quality_ok"] is True
    assert qwen_row["graph_prior_recall_at_k"] == pytest.approx(1.0)
    assert qwen_row["graph_prior_delta_recovery"] > 0.0
    assert random_rows
    assert all("graph_prior_recall_at_k" in row for row in random_rows)
    assert all("graph_prior_precision_at_k" in row for row in random_rows)
    assert all(row["graph_prior_quality_ok"] is False for row in random_rows)

    promotion = json.loads((out / "promotion_decision.json").read_text())
    assert promotion["promote"] is False
    assert promotion["quality_ok"] is False
    assert promotion["memory_ok"] is False
    assert promotion["speed_ok"] is False


def test_qwen_energy_capture_beats_matched_random_on_non_implanted_fixture(tmp_path):
    artifact_dir = _compile_energy_proxy_fixture(tmp_path)
    out = tmp_path / "prior_eval_energy"
    result = run_graph_prior_eval(
        source_weight_graph_dir=artifact_dir,
        output_dir=out,
        k_values=[1, 2],
        random_seeds=[0, 1, 2, 3, 4],
        quality_mode="energy_capture",
    )

    report = result["graph_prior_quality_report"]
    assert report is not None
    assert (out / "graph_prior_quality_report.json").exists()
    assert report["status"] == "energy_capture_proxy"
    assert report["quality_ok"] is True
    assert report["delta_energy_capture_ratio"] > 0.0
    assert report["qwen_energy_capture_ratio"] > report["random_energy_capture_ratio_mean"]

    qwen_row = next(row for row in result["baseline_matrix"] if row["adjacency_name"] == "qwen_topk_k1")
    random_rows = [
        row
        for row in result["baseline_matrix"]
        if row["adjacency_name"].startswith("random_matched_k1_seed")
    ]
    assert qwen_row["graph_prior_metric"] == "energy_capture_proxy"
    assert qwen_row["graph_prior_quality_ok"] is True
    assert qwen_row["quality_ok"] is True
    assert qwen_row["graph_prior_delta_energy_capture_ratio"] > 0.0
    assert random_rows
    assert all("graph_prior_energy_capture" in row for row in random_rows)
    assert all("graph_prior_energy_capture_ratio" in row for row in random_rows)
    assert all(row["graph_prior_quality_ok"] is False for row in random_rows)
    assert result["promotion_decision"]["promote"] is False
    assert result["promotion_decision"]["quality_ok"] is False


def test_graph_prior_eval_writes_selected_adjacency_handoff_artifacts(tmp_path):
    artifact_dir = _compile_energy_proxy_fixture(tmp_path)
    out = tmp_path / "prior_eval_handoff"
    result = run_graph_prior_eval(
        source_weight_graph_dir=artifact_dir,
        output_dir=out,
        k_values=[1, 2],
        random_seeds=[0, 1],
        quality_mode="energy_capture",
    )

    selected_dir = out / "selected_adjacencies"
    index_path = selected_dir / "index.json"
    k1_path = selected_dir / "qwen_topk_k1.json"
    k2_path = selected_dir / "qwen_topk_k2.json"
    handoff_path = out / "v25_handoff_manifest.json"
    assert index_path.exists()
    assert k1_path.exists()
    assert k2_path.exists()
    assert handoff_path.exists()

    index = json.loads(index_path.read_text())
    assert index == result["selected_adjacency_index"]
    assert index["schema_version"] == "qwen_selected_adjacency_index.v1"
    assert index["bounded"] is True
    assert len(index["adjacencies"]) == 2
    assert {row["adjacency_name"] for row in index["adjacencies"]} == {"qwen_topk_k1", "qwen_topk_k2"}

    selected = json.loads(k1_path.read_text())
    assert selected["schema_version"] == "qwen_selected_adjacency.v1"
    assert selected["adjacency_name"] == "qwen_topk_k1"
    assert selected["bounded"] is True
    assert selected["source"] == "G_0"
    assert selected["selection_policy"] == "per_source_topk_score_desc"
    assert selected["edge_score_name"] == "normalized_frobenius"
    assert selected["graph_scope"] == "attention_mlp_moe"
    assert selected["edges"]
    for edge in selected["edges"]:
        assert {"edge_id", "src_id", "dst_id", "relation", "weight", "score_name"}.issubset(edge)
        assert "raw_weight_payload" not in edge
        assert "tensor_values" not in edge
        assert "values" not in edge
        assert edge["source"] == "G_0"

    handoff = json.loads(handoff_path.read_text())
    assert handoff == result["v25_handoff_manifest"]
    assert handoff["schema_version"] == "qwen_v25_handoff.v1"
    assert handoff["teacher_checkpoint_loaded"] is False
    assert handoff["raw_weight_payload_in_graph"] is False
    assert handoff["bounded_active_adjacency"] is True
    assert handoff["student_training_started"] is False
    assert handoff["promotion_required_before_deploy"] is True
    assert handoff["selected_adjacency_index"] == "selected_adjacencies/index.json"
    assert handoff["graph_prior_quality_report"] == "graph_prior_quality_report.json"

    prior_config = json.loads((out / "prior_config.json").read_text())
    assert prior_config["selected_adjacency_index"] == "selected_adjacencies/index.json"
    assert prior_config["v25_handoff_manifest"] == "v25_handoff_manifest.json"
    assert result["promotion_decision"]["promote"] is False
    assert result["promotion_decision"]["quality_ok"] is False


def test_graph_prior_eval_cli_artifact_contract_energy_capture(tmp_path):
    artifact_dir = _compile_energy_proxy_fixture(tmp_path)
    out = tmp_path / "prior_eval_energy_cli"
    project_root = Path(__file__).resolve().parents[1]

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.qwen_graph_prior_eval",
            "--source-weight-graph-dir",
            str(artifact_dir),
            "--output-dir",
            str(out),
            "--k-values",
            "1,2",
            "--random-seeds",
            "0,1,2,3,4",
            "--graph-scope",
            "attention_mlp_moe",
            "--quality-mode",
            "energy_capture",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    for name in (
        "graph_prior_quality_report.json",
        "baseline_matrix.json",
        "quality_report.json",
        "promotion_decision.json",
        "v25_handoff_manifest.json",
    ):
        assert (out / name).exists()
    assert (out / "selected_adjacencies" / "index.json").exists()
    assert (out / "selected_adjacencies" / "qwen_topk_k1.json").exists()
    assert (out / "selected_adjacencies" / "qwen_topk_k2.json").exists()

    report = json.loads((out / "graph_prior_quality_report.json").read_text())
    assert report["status"] == "energy_capture_proxy"
    assert report["quality_ok"] is True
    assert report["delta_energy_capture_ratio"] > 0.0

    matrix = json.loads((out / "baseline_matrix.json").read_text())
    qwen_row = next(row for row in matrix if row["adjacency_name"] == "qwen_topk_k1")
    random_rows = [row for row in matrix if row["adjacency_name"].startswith("random_matched_k1_seed")]
    assert qwen_row["graph_prior_metric"] == "energy_capture_proxy"
    assert qwen_row["graph_prior_quality_ok"] is True
    assert qwen_row["graph_prior_energy_capture_ratio"] > 0.0
    assert random_rows
    assert all("graph_prior_energy_capture" in row for row in random_rows)
    assert all(row["graph_prior_quality_ok"] is False for row in random_rows)

    quality_report = json.loads((out / "quality_report.json").read_text())
    assert quality_report["status"] == "graph_prior_quality_run"
    assert quality_report["graph_prior_quality_ok"] is True

    promotion = json.loads((out / "promotion_decision.json").read_text())
    assert promotion["promote"] is False
    assert promotion["quality_ok"] is False
    assert promotion["memory_ok"] is False
    assert promotion["speed_ok"] is False

    handoff = json.loads((out / "v25_handoff_manifest.json").read_text())
    assert handoff["teacher_checkpoint_loaded"] is False
    assert handoff["raw_weight_payload_in_graph"] is False
    assert handoff["bounded_active_adjacency"] is True
    assert handoff["student_training_started"] is False
    assert handoff["promotion_required_before_deploy"] is True


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
