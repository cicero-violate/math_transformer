from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.v26_rewire_k2_cycle import (
    SCHEMA_VERSION,
    CYCLE_REPORT_FILENAME,
    build_synthetic_edge_trace_dir,
    run_v26_rewire_cycle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAMILIES = ["arithmetic_short", "logic_short", "symbolic_short", "project_specific"]


def _make_adjacency(name: str = "qwen_topk_k2", k: int = 2, n_src: int = 4) -> dict:
    edges = []
    idx = 0
    for src_i in range(n_src):
        for dst_offset in range(1, k + 1):
            dst_i = (src_i + dst_offset) % (n_src + k)
            edges.append({
                "edge_id": f"e{idx:04d}",
                "src_id": f"src{src_i}",
                "dst_id": f"dst{dst_i}",
                "weight": 0.9 - idx * 0.01,
                "relation": "qk_affinity_prior",
                "score_name": "normalized_frobenius",
                "source": "G_0",
                "metadata": {"provenance": {"block_in": 0, "block_out": 0, "shard": "model.safetensors"}, "source_tensor": "model.layers.0.self_attn.k_proj.weight"},
            })
            idx += 1
    node_count = len({str(e["src_id"]) for e in edges} | {str(e["dst_id"]) for e in edges})
    return {
        "schema_version": "qwen_selected_adjacency.v1",
        "adjacency_name": name,
        "k": k,
        "bounded": True,
        "source": "G_0",
        "selection_policy": "per_source_topk_score_desc",
        "node_count": node_count,
        "edge_count": len(edges),
        "edge_score_name": "normalized_frobenius",
        "edges": edges,
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _make_handoff_dir(out: Path, adjacency: dict, weight_graph_dir: Path) -> Path:
    """Create a minimal v25 handoff dir with the given adjacency."""
    adj_dir = out / "selected_adjacencies"
    adj_dir.mkdir(parents=True, exist_ok=True)
    name = adjacency["adjacency_name"]
    _write_json(adj_dir / f"{name}.json", adjacency)
    index = {
        "schema_version": "qwen_selected_adjacency_index.v1",
        "bounded": True,
        "selection_policy": "per_source_topk_score_desc",
        "edge_score_name": "normalized_frobenius",
        "graph_scope": "attention_mlp_moe",
        "source_weight_graph_dir": str(weight_graph_dir),
        "adjacencies": [{
            "adjacency_name": name,
            "k": adjacency["k"],
            "edge_count": adjacency["edge_count"],
            "node_count": adjacency["node_count"],
            "path": f"selected_adjacencies/{name}.json",
        }],
    }
    _write_json(adj_dir / "index.json", index)
    handoff = {
        "schema_version": "qwen_v25_handoff.v1",
        "status": "ready_for_fixed_topology_sparse_student",
        "source_weight_graph_dir": str(weight_graph_dir),
        "selected_adjacency_index": "selected_adjacencies/index.json",
        "teacher_checkpoint_loaded": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "promotion_required_before_deploy": True,
        "student_training_started": False,
        "quality_mode": "energy_capture",
        "graph_prior_quality_report": None,
        "promotion_decision": None,
    }
    _write_json(out / "v25_handoff_manifest.json", handoff)
    prior_config = {
        "schema_version": "qwen_graph_prior_eval.v1",
        "graph_scope": "attention_mlp_moe",
        "edge_score_name": "normalized_frobenius",
        "source_weight_graph_dir": str(weight_graph_dir),
        "selected_adjacency_index": "selected_adjacencies/index.json",
        "v25_handoff_manifest": "v25_handoff_manifest.json",
        "quality_mode": "energy_capture",
        "teacher_checkpoint_loaded": False,
    }
    _write_json(out / "prior_config.json", prior_config)
    return out


def _make_weight_graph_dir(out: Path, adjacency: dict) -> Path:
    """Create a minimal weight graph dir with extra candidate edges."""
    edges = []
    idx = 0
    all_nodes = list({str(e["src_id"]) for e in adjacency["edges"]}
                     | {str(e["dst_id"]) for e in adjacency["edges"]})
    all_src = sorted({str(e["src_id"]) for e in adjacency["edges"]})
    # Include the existing edges
    for e in adjacency["edges"]:
        edges.append({
            "edge_id": e["edge_id"],
            "src": e["src_id"],
            "rel": "qk_affinity",
            "dst": e["dst_id"],
            "weight": float(e["weight"]),
            "score_name": "normalized_frobenius",
            "source_tensor": "model.layers.0.self_attn.k_proj.weight",
            "provenance": {"layer": 0},
        })
        idx += 1
    # Extra candidate edges (alternatives not in current adjacency)
    extra_weight = 0.95
    for src in all_src:
        for dst in all_nodes:
            existing_dsts = {str(e["dst_id"]) for e in adjacency["edges"] if str(e["src_id"]) == src}
            if dst not in existing_dsts and dst != src:
                edges.append({
                    "edge_id": f"cand_{idx:04d}",
                    "src": src,
                    "rel": "qk_affinity",
                    "dst": dst,
                    "weight": extra_weight,
                    "score_name": "normalized_frobenius",
                    "source_tensor": "model.layers.0.self_attn.k_proj.weight",
                    "provenance": {"layer": 0},
                })
                idx += 1
                extra_weight -= 0.001
    nodes = [{"node_id": n, "type": "channel_block", "label": n, "layer": 0} for n in all_nodes]
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "edges.jsonl", edges)
    _write_jsonl(out / "nodes.jsonl", nodes)
    _write_json(out / "manifest.json", {
        "schema_version": "qwen_weight_graph.v1",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "graph_scope": "attention_mlp_moe",
    })
    _write_json(out / "stats.json", {
        "node_count": len(nodes),
        "edge_count": len(edges),
    })
    return out


def _make_distill_examples(path: Path, n_per_family: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for fam in FAMILIES:
            for i in range(n_per_family):
                row = {
                    "sample_id": f"{fam}_{i}",
                    "family": fam,
                    "input": f"question {i}",
                    "target": f"reasoning: compute {i}.\nanswer: {i % 3}",
                    "split": "train",
                }
                fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# build_synthetic_edge_trace_dir
# ---------------------------------------------------------------------------

def test_synthetic_edge_trace_schema(tmp_path):
    adj = _make_adjacency()
    trace_dir = build_synthetic_edge_trace_dir(adj, tmp_path / "trace")
    report = json.loads((trace_dir / "edge_trace_report.json").read_text())
    assert report["schema_version"] == "qwen_edge_trace.v1"
    assert report["status"] == "edge_trace_ok"
    assert report["adjacency_name"] == "qwen_topk_k2"
    assert report["k"] == 2
    assert report["edge_count"] == adj["edge_count"]
    assert report["bounded_active_adjacency"] is True
    assert report["topology_mutated"] is False
    assert report["finite"] is True


def test_synthetic_edge_trace_utility_summary(tmp_path):
    adj = _make_adjacency(n_src=3)
    trace_dir = build_synthetic_edge_trace_dir(adj, tmp_path / "trace")
    report = json.loads((trace_dir / "edge_trace_report.json").read_text())
    summary = report["edge_utility_summary"]
    assert summary["schema_version"] == "qwen_edge_utility_summary.v1"
    assert len(summary["ranked_edges"]) == adj["edge_count"]
    for row in summary["ranked_edges"]:
        assert "edge_id" in row
        assert row["utility_score"] == 0.0


def test_synthetic_edge_trace_validates(tmp_path):
    from src.qwen_edge_trace import validate_edge_trace_report, load_edge_trace_report
    adj = _make_adjacency()
    trace_dir = build_synthetic_edge_trace_dir(adj, tmp_path / "trace")
    report = load_edge_trace_report(trace_dir)
    summary = validate_edge_trace_report(report)
    assert summary["status"] == "edge_trace_report_valid"


# ---------------------------------------------------------------------------
# run_v26_rewire_cycle (integration — uses the real teacher artifacts dir)
# ---------------------------------------------------------------------------

def test_run_v26_rewire_cycle_basic(tmp_path):
    adjacency = _make_adjacency(n_src=4)
    weight_graph_dir = _make_weight_graph_dir(tmp_path / "wg", adjacency)
    handoff_dir = _make_handoff_dir(tmp_path / "handoff", adjacency, weight_graph_dir)
    artifacts_dir = tmp_path / "teacher_artifacts"
    _make_distill_examples(artifacts_dir / "distill_examples.jsonl", n_per_family=6)

    report = run_v26_rewire_cycle(
        handoff_dir,
        tmp_path / "out",
        artifacts_dir,
        baseline_k=2,
        baseline_adjacency_name="qwen_topk_k2",
        max_swaps=2,
        proposal_policy="same_source_top_weight",
        vocab_size=8,
        feature_dim=4,
        forward_steps=1,
        train_steps=3,
        lr=0.1,
        held_out_per_family=2,
        heldout_train_steps=4,
        heldout_lr=0.5,
    )
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["status"] == "v26_rewire_k2_cycle_ok"
    assert report["baseline_adjacency"] == "qwen_topk_k2"
    assert report["k"] == 2
    assert math.isfinite(report["kl_baseline_after"])
    assert math.isfinite(report["kl_candidate_after"])
    assert math.isfinite(report["kl_random_after"])
    assert math.isfinite(report["heldout_loss_mean"])
    assert report["finite"] is True
    assert report["teacher_checkpoint_loaded"] is False
    assert report["raw_weight_payload_in_graph"] is False
    assert report["bounded_active_adjacency"] is True
    assert report["topology_mutated"] is False
    assert isinstance(report["promote"], bool)
    assert isinstance(report["quality_ok"], bool)
    assert isinstance(report["kl_ok"], bool)
    assert isinstance(report["heldout_ok"], bool)


def test_run_v26_rewire_cycle_writes_report_file(tmp_path):
    adjacency = _make_adjacency(n_src=3)
    weight_graph_dir = _make_weight_graph_dir(tmp_path / "wg", adjacency)
    handoff_dir = _make_handoff_dir(tmp_path / "handoff", adjacency, weight_graph_dir)
    artifacts_dir = tmp_path / "teacher_artifacts"
    _make_distill_examples(artifacts_dir / "distill_examples.jsonl", n_per_family=5)

    run_v26_rewire_cycle(
        handoff_dir,
        tmp_path / "out",
        artifacts_dir,
        vocab_size=8,
        feature_dim=4,
        train_steps=2,
        held_out_per_family=2,
        heldout_train_steps=4,
    )
    assert (tmp_path / "out" / CYCLE_REPORT_FILENAME).exists()


def test_run_v26_rewire_cycle_safety_flags(tmp_path):
    adjacency = _make_adjacency(n_src=3)
    weight_graph_dir = _make_weight_graph_dir(tmp_path / "wg", adjacency)
    handoff_dir = _make_handoff_dir(tmp_path / "handoff", adjacency, weight_graph_dir)
    artifacts_dir = tmp_path / "teacher_artifacts"
    _make_distill_examples(artifacts_dir / "distill_examples.jsonl", n_per_family=4)

    report = run_v26_rewire_cycle(
        handoff_dir, tmp_path / "out", artifacts_dir,
        vocab_size=4, feature_dim=4, train_steps=2,
        held_out_per_family=1, heldout_train_steps=4,
    )
    assert report["teacher_checkpoint_loaded"] is False
    assert report["raw_weight_payload_in_graph"] is False
    assert report["bounded_active_adjacency"] is True
    assert report["topology_mutated"] is False


def test_run_v26_rewire_cycle_kl_delta_fields(tmp_path):
    adjacency = _make_adjacency(n_src=4)
    weight_graph_dir = _make_weight_graph_dir(tmp_path / "wg", adjacency)
    handoff_dir = _make_handoff_dir(tmp_path / "handoff", adjacency, weight_graph_dir)
    artifacts_dir = tmp_path / "teacher_artifacts"
    _make_distill_examples(artifacts_dir / "distill_examples.jsonl", n_per_family=6)

    report = run_v26_rewire_cycle(
        handoff_dir, tmp_path / "out", artifacts_dir,
        vocab_size=8, feature_dim=4, train_steps=3,
        held_out_per_family=2, heldout_train_steps=4,
    )
    assert abs(report["kl_delta_vs_baseline"] - (report["kl_candidate_after"] - report["kl_baseline_after"])) < 1e-9
    assert abs(report["kl_delta_vs_random"] - (report["kl_candidate_after"] - report["kl_random_after"])) < 1e-9
    assert report["candidate_beats_baseline"] == (report["kl_candidate_after"] < report["kl_baseline_after"])
    assert report["candidate_beats_random"] == (report["kl_candidate_after"] < report["kl_random_after"])
    assert report["kl_ok"] == (report["candidate_beats_baseline"] and report["candidate_beats_random"])
    assert report["promote"] == (report["quality_ok"] and report["kl_ok"] and report["heldout_ok"])
