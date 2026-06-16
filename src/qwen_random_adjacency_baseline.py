from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from src.qwen_kl_distillation_eval import run_and_write_kl_eval_report
from src.qwen_logit_distillation_targets import (
    validate_frozen_logit_distillation_targets,
    write_frozen_logit_distillation_targets,
)
from src.qwen_sparse_student_handoff import (
    HANDOFF_SCHEMA_VERSION,
    SELECTED_ADJACENCY_SCHEMA_VERSION,
    SELECTED_INDEX_SCHEMA_VERSION,
    load_selected_adjacency,
    validate_selected_adjacency,
)


SCHEMA_VERSION = "qwen_random_adjacency_baseline.v1"
BASELINE_REPORT_FILENAME = "baseline_comparison_report.json"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_matched_random_adjacency(
    qwen_adjacency: dict[str, Any],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Randomize edge destinations while preserving: node universe, per-source out-degrees, k.

    Guarantees every node in the original node set appears as an endpoint:
    - Non-source nodes (never a src_id) are assigned as mandatory destinations first.
    - Remaining destination slots get random non-self-loop assignments.
    """
    edges = qwen_adjacency["edges"]
    k = int(qwen_adjacency["k"])
    base_name = str(qwen_adjacency["adjacency_name"])

    all_node_ids = sorted(
        {str(e["src_id"]) for e in edges} | {str(e["dst_id"]) for e in edges}
    )
    src_counts: Counter[str] = Counter(str(e["src_id"]) for e in edges)
    src_set = set(src_counts.keys())

    # Non-source nodes must appear as destinations (guaranteed self-loop-free since
    # non-source nodes are never equal to a src_id).
    mandatory_dsts: list[str] = sorted(set(all_node_ids) - src_set)

    rng = random.Random(seed)

    # Build flat (src_id, slot_index) list and shuffle to randomize which src gets
    # which mandatory destination.
    src_slots: list[tuple[str, int]] = []
    for src_id in sorted(src_counts.keys()):
        for slot_idx in range(src_counts[src_id]):
            src_slots.append((src_id, slot_idx))
    rng.shuffle(src_slots)

    mandatory_queue: list[str] = list(mandatory_dsts)
    rng.shuffle(mandatory_queue)

    slot_dst: dict[tuple[str, int], str] = {}
    for src_id, slot_idx in src_slots:
        if mandatory_queue:
            # Mandatory dsts are never src_id (non-source nodes), so always safe.
            slot_dst[(src_id, slot_idx)] = mandatory_queue.pop(0)
        else:
            candidates = [nid for nid in all_node_ids if nid != src_id]
            slot_dst[(src_id, slot_idx)] = rng.choice(candidates if candidates else all_node_ids)

    new_edges: list[dict[str, Any]] = []
    edge_idx = 0
    for src_id in sorted(src_counts.keys()):
        for slot_idx in range(src_counts[src_id]):
            new_edges.append(
                {
                    "edge_id": f"random_e{edge_idx}",
                    "src_id": src_id,
                    "dst_id": slot_dst[(src_id, slot_idx)],
                    "relation": "random_baseline",
                    "weight": 1.0,
                    "score_name": "random",
                }
            )
            edge_idx += 1

    actual_node_ids = sorted(
        {str(e["src_id"]) for e in new_edges} | {str(e["dst_id"]) for e in new_edges}
    )
    if set(actual_node_ids) != set(all_node_ids):
        raise ValueError(
            f"random adjacency node universe mismatch: "
            f"got {len(actual_node_ids)}, expected {len(all_node_ids)}"
        )

    random_name = f"{base_name}_random_seed{seed}"
    return {
        "schema_version": SELECTED_ADJACENCY_SCHEMA_VERSION,
        "adjacency_name": random_name,
        "k": k,
        "bounded": True,
        "source": "G_0",
        "selection_policy": "random_matched_baseline",
        "node_count": len(actual_node_ids),
        "edge_count": len(new_edges),
        "edges": new_edges,
    }


def _build_handoff_dir(
    out_dir: Path,
    *,
    qwen_adjacency: dict[str, Any],
    random_adjacency: dict[str, Any],
    qwen_summary: dict[str, Any],
    random_summary: dict[str, Any],
    source_eval_dir: Path,
) -> None:
    adj_dir = out_dir / "selected_adjacencies"
    adj_dir.mkdir(parents=True, exist_ok=True)

    _write_json(adj_dir / f"{qwen_summary['adjacency_name']}.json", qwen_adjacency)
    _write_json(adj_dir / f"{random_summary['adjacency_name']}.json", random_adjacency)

    index = {
        "schema_version": SELECTED_INDEX_SCHEMA_VERSION,
        "bounded": True,
        "selection_policy": "mixed_qwen_and_random_baseline",
        "adjacencies": [
            {
                "adjacency_name": qwen_summary["adjacency_name"],
                "k": qwen_summary["k"],
                "edge_count": qwen_summary["edge_count"],
                "node_count": qwen_summary["node_count"],
                "path": f"selected_adjacencies/{qwen_summary['adjacency_name']}.json",
            },
            {
                "adjacency_name": random_summary["adjacency_name"],
                "k": random_summary["k"],
                "edge_count": random_summary["edge_count"],
                "node_count": random_summary["node_count"],
                "path": f"selected_adjacencies/{random_summary['adjacency_name']}.json",
            },
        ],
    }
    _write_json(adj_dir / "index.json", index)

    handoff = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "status": "ready_for_fixed_topology_sparse_student",
        "teacher_checkpoint_loaded": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "student_training_started": False,
        "promotion_required_before_deploy": True,
        "selected_adjacency_index": "selected_adjacencies/index.json",
        "source_weight_graph_dir": str(source_eval_dir),
    }
    _write_json(out_dir / "v25_handoff_manifest.json", handoff)


def run_kl_baseline_comparison(
    source_eval_dir: str | Path,
    output_dir: str | Path,
    *,
    k: int | None = None,
    adjacency_name: str | None = None,
    random_seed: int = 0,
    vocab_size: int = 16,
    target_seeds: list[int] | None = None,
    feature_dim: int = 8,
    forward_steps: int = 1,
    projection_seed: int = 0,
    temperature: float = 1.0,
    device: str = "cpu",
) -> dict[str, Any]:
    source_dir = Path(source_eval_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qwen_adjacency = load_selected_adjacency(source_dir, adjacency_name=adjacency_name, k=k)
    qwen_summary = validate_selected_adjacency(qwen_adjacency)

    random_adjacency = generate_matched_random_adjacency(qwen_adjacency, seed=random_seed)
    random_summary = validate_selected_adjacency(random_adjacency)

    _build_handoff_dir(
        out_dir,
        qwen_adjacency=qwen_adjacency,
        random_adjacency=random_adjacency,
        qwen_summary=qwen_summary,
        random_summary=random_summary,
        source_eval_dir=source_dir,
    )

    targets_dir = out_dir / "logit_targets"
    seeds = target_seeds if target_seeds is not None else list(range(8))
    write_frozen_logit_distillation_targets(
        targets_dir,
        vocab_size=vocab_size,
        seeds=seeds,
        temperature=temperature,
    )
    validate_frozen_logit_distillation_targets(targets_dir)

    qwen_eval = run_and_write_kl_eval_report(
        out_dir,
        targets_dir,
        out_dir / "kl_eval_qwen.json",
        adjacency_name=qwen_summary["adjacency_name"],
        feature_dim=feature_dim,
        steps=forward_steps,
        projection_seed=projection_seed,
        temperature=temperature,
        device=device,
    )

    random_eval = run_and_write_kl_eval_report(
        out_dir,
        targets_dir,
        out_dir / "kl_eval_random.json",
        adjacency_name=random_summary["adjacency_name"],
        feature_dim=feature_dim,
        steps=forward_steps,
        projection_seed=projection_seed,
        temperature=temperature,
        device=device,
    )

    kl_qwen = float(qwen_eval["kl_mean"])
    kl_random = float(random_eval["kl_mean"])
    delta = kl_qwen - kl_random
    qwen_wins = kl_qwen < kl_random

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "baseline_comparison_ok",
        "source_eval_dir": str(source_dir),
        "output_dir": str(out_dir),
        "qwen_adjacency_name": qwen_summary["adjacency_name"],
        "random_adjacency_name": random_summary["adjacency_name"],
        "k": qwen_summary["k"],
        "node_count": qwen_summary["node_count"],
        "edge_count_qwen": qwen_summary["edge_count"],
        "edge_count_random": random_summary["edge_count"],
        "random_seed": random_seed,
        "vocab_size": vocab_size,
        "feature_dim": feature_dim,
        "forward_steps": forward_steps,
        "temperature": temperature,
        "device": device,
        "kl_mean_qwen": kl_qwen,
        "kl_mean_random": kl_random,
        "kl_delta_qwen_minus_random": delta,
        "qwen_topology_wins": qwen_wins,
        "finite": math.isfinite(kl_qwen) and math.isfinite(kl_random),
        "note": (
            "lower KL is better; qwen_topology_wins=true means the Qwen weight-graph "
            "topology produces lower KL than the matched random baseline"
        ),
    }
    _write_json(out_dir / BASELINE_REPORT_FILENAME, report)
    return report
