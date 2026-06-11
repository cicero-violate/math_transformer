from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

from .qwen_graph_prior_eval import (
    PriorAdjacency,
    build_matched_random_adjacency,
    build_qwen_topk_adjacency,
)
from .qwen_weight_graph import WeightGraphResult, read_weight_graph_artifacts


SCHEMA_VERSION = "qwen_graph_prior_quality.v1"
EdgeKey = tuple[str, str, str]


@dataclass(frozen=True)
class RecoveryMetrics:
    recall_at_k: float
    precision_at_k: float
    hit_count: int
    pred_count: int
    gold_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "hit_count": self.hit_count,
            "pred_count": self.pred_count,
            "gold_count": self.gold_count,
        }


@dataclass(frozen=True)
class EnergyCaptureMetrics:
    energy_capture: float
    energy_capture_ratio: float
    hit_count: int
    pred_count: int
    candidate_edge_count: int
    candidate_energy_total: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "energy_capture": self.energy_capture,
            "energy_capture_ratio": self.energy_capture_ratio,
            "hit_count": self.hit_count,
            "pred_count": self.pred_count,
            "candidate_edge_count": self.candidate_edge_count,
            "candidate_energy_total": self.candidate_energy_total,
        }


def edge_key(src_id: str, relation: str, dst_id: str) -> EdgeKey:
    return (str(src_id), str(relation), str(dst_id))


def adjacency_edge_keys(adjacency: PriorAdjacency) -> set[EdgeKey]:
    return {edge_key(edge.src_id, edge.relation, edge.dst_id) for edge in adjacency.edges}


def recovery_metrics(adjacency: PriorAdjacency, gold_edges: Iterable[EdgeKey]) -> RecoveryMetrics:
    gold = set(gold_edges)
    pred = adjacency_edge_keys(adjacency)
    hits = pred & gold
    recall = float(len(hits)) / float(len(gold)) if gold else 0.0
    precision = float(len(hits)) / float(len(pred)) if pred else 0.0
    return RecoveryMetrics(
        recall_at_k=recall,
        precision_at_k=precision,
        hit_count=len(hits),
        pred_count=len(pred),
        gold_count=len(gold),
    )


def candidate_energy_by_edge_key(candidate_edges: Iterable[Any]) -> dict[EdgeKey, float]:
    """
    Build the G0 energy table used by the non-implanted proxy.

    Random matched edges are synthetic. They only capture energy when their
    sampled (src, relation, dst) key exists in this G0 candidate table.
    """
    energy: dict[EdgeKey, float] = {}
    for edge in candidate_edges:
        src_id = getattr(edge, "src_id")
        relation = getattr(edge, "relation", getattr(edge, "rel", ""))
        dst_id = getattr(edge, "dst_id")
        weight = max(0.0, float(getattr(edge, "weight", 0.0)))
        key = edge_key(src_id, relation, dst_id)
        energy[key] = max(float(energy.get(key, 0.0)), weight)
    return energy


def energy_capture_metrics(
    adjacency: PriorAdjacency,
    candidate_energy: dict[EdgeKey, float],
) -> EnergyCaptureMetrics:
    pred = adjacency_edge_keys(adjacency)
    captured = sum(float(candidate_energy.get(key, 0.0)) for key in pred)
    total = sum(float(v) for v in candidate_energy.values())
    ratio = captured / total if total > 0.0 else 0.0
    return EnergyCaptureMetrics(
        energy_capture=captured,
        energy_capture_ratio=ratio,
        hit_count=sum(1 for key in pred if candidate_energy.get(key, 0.0) > 0.0),
        pred_count=len(pred),
        candidate_edge_count=len(candidate_energy),
        candidate_energy_total=total,
    )


def evaluate_prior_energy_capture(
    *,
    qwen_adjacency: PriorAdjacency,
    random_adjacencies: list[PriorAdjacency],
    candidate_edges: Iterable[Any],
) -> dict[str, Any]:
    candidate_energy = candidate_energy_by_edge_key(candidate_edges)
    qwen_metrics = energy_capture_metrics(qwen_adjacency, candidate_energy)
    random_metrics = [energy_capture_metrics(adj, candidate_energy) for adj in random_adjacencies]
    random_energy = [m.energy_capture for m in random_metrics]
    random_ratio = [m.energy_capture_ratio for m in random_metrics]
    energy_mean, energy_std = _random_stats(random_energy)
    ratio_mean, ratio_std = _random_stats(random_ratio)
    delta_energy = qwen_metrics.energy_capture - energy_mean
    delta_ratio = qwen_metrics.energy_capture_ratio - ratio_mean
    quality_ok = delta_ratio > 0.0 and qwen_metrics.energy_capture_ratio > ratio_mean
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "energy_capture_proxy",
        "adjacency_name": qwen_adjacency.name,
        "k": qwen_adjacency.k,
        "quality_ok": quality_ok,
        "qwen_energy_capture": qwen_metrics.energy_capture,
        "qwen_energy_capture_ratio": qwen_metrics.energy_capture_ratio,
        "qwen_hit_count": qwen_metrics.hit_count,
        "qwen_pred_count": qwen_metrics.pred_count,
        "random_energy_capture_mean": energy_mean,
        "random_energy_capture_std": energy_std,
        "random_energy_capture_ratio_mean": ratio_mean,
        "random_energy_capture_ratio_std": ratio_std,
        "random_seed_count": len(random_adjacencies),
        "delta_energy_capture": delta_energy,
        "delta_energy_capture_ratio": delta_ratio,
        "candidate_edge_count": qwen_metrics.candidate_edge_count,
        "candidate_energy_total": qwen_metrics.candidate_energy_total,
        "random_rows": [
            {
                "adjacency_name": adj.name,
                "energy_capture": metrics.energy_capture,
                "energy_capture_ratio": metrics.energy_capture_ratio,
                "hit_count": metrics.hit_count,
                "pred_count": metrics.pred_count,
            }
            for adj, metrics in zip(random_adjacencies, random_metrics)
        ],
    }


def gold_edges_from_block_specs(
    result: WeightGraphResult,
    specs: Iterable[dict[str, Any]],
    *,
    score_name: str = "normalized_frobenius",
) -> set[EdgeKey]:
    """
    Resolve fixture gold blocks to compiled graph edge keys.

    Each spec should include source_tensor, block_in, block_out, and relation.
    This keeps tests and artifacts free of raw tensor values.
    """
    wanted = {
        (
            str(spec["source_tensor"]),
            int(spec["block_in"]),
            int(spec["block_out"]),
            str(spec["relation"]),
        )
        for spec in specs
    }
    found: set[EdgeKey] = set()
    for edge in result.edges:
        provenance = edge.provenance or {}
        key = (
            str(edge.source_tensor),
            int(provenance.get("block_in", -1)),
            int(provenance.get("block_out", -1)),
            str(edge.rel),
        )
        if edge.score_name == score_name and key in wanted:
            found.add(edge_key(edge.src_id, edge.rel, edge.dst_id))
    missing = wanted - {
        (
            str(edge.source_tensor),
            int((edge.provenance or {}).get("block_in", -1)),
            int((edge.provenance or {}).get("block_out", -1)),
            str(edge.rel),
        )
        for edge in result.edges
        if edge.score_name == score_name
    }
    if missing:
        raise ValueError(f"gold block specs not found in compiled graph: {sorted(missing)}")
    return found


def _random_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(mean(values)), float(pstdev(values))


def evaluate_prior_recovery(
    *,
    qwen_adjacency: PriorAdjacency,
    random_adjacencies: list[PriorAdjacency],
    gold_edges: Iterable[EdgeKey],
    min_qwen_recall: float = 0.95,
) -> dict[str, Any]:
    gold = set(gold_edges)
    qwen_metrics = recovery_metrics(qwen_adjacency, gold)
    random_metrics = [recovery_metrics(adj, gold) for adj in random_adjacencies]
    random_recalls = [m.recall_at_k for m in random_metrics]
    random_precisions = [m.precision_at_k for m in random_metrics]
    recall_mean, recall_std = _random_stats(random_recalls)
    precision_mean, precision_std = _random_stats(random_precisions)
    delta = qwen_metrics.recall_at_k - recall_mean
    quality_ok = delta > 0.0 and qwen_metrics.recall_at_k >= min_qwen_recall
    return {
        "schema_version": SCHEMA_VERSION,
        "adjacency_name": qwen_adjacency.name,
        "k": qwen_adjacency.k,
        "gold_edge_count": len(gold),
        "qwen_recall_at_k": qwen_metrics.recall_at_k,
        "qwen_precision_at_k": qwen_metrics.precision_at_k,
        "qwen_hit_count": qwen_metrics.hit_count,
        "qwen_pred_count": qwen_metrics.pred_count,
        "random_recall_mean": recall_mean,
        "random_recall_std": recall_std,
        "random_precision_mean": precision_mean,
        "random_precision_std": precision_std,
        "random_seed_count": len(random_adjacencies),
        "delta_recovery": delta,
        "min_qwen_recall": min_qwen_recall,
        "quality_ok": quality_ok,
        "random_rows": [
            {
                "adjacency_name": adj.name,
                "recall_at_k": metrics.recall_at_k,
                "precision_at_k": metrics.precision_at_k,
                "hit_count": metrics.hit_count,
                "pred_count": metrics.pred_count,
            }
            for adj, metrics in zip(random_adjacencies, random_metrics)
        ],
    }


def write_quality_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_prior_recovery_quality(
    *,
    source_weight_graph_dir: str | Path,
    gold_block_specs: Iterable[dict[str, Any]],
    output_path: str | Path,
    k: int = 1,
    random_seeds: list[int] | None = None,
    graph_scope: str = "attention_mlp_moe",
    edge_score_name: str = "normalized_frobenius",
    min_qwen_recall: float = 0.95,
) -> dict[str, Any]:
    result = read_weight_graph_artifacts(source_weight_graph_dir)
    qwen = build_qwen_topk_adjacency(
        result,
        k=k,
        graph_scope=graph_scope,
        edge_score_name=edge_score_name,
    )
    seeds = list(random_seeds or [0, 1, 2, 3, 4])
    random_adjacencies = [
        build_matched_random_adjacency(qwen, result, seed=seed)
        for seed in seeds
    ]
    gold_edges = gold_edges_from_block_specs(result, gold_block_specs, score_name=edge_score_name)
    report = evaluate_prior_recovery(
        qwen_adjacency=qwen,
        random_adjacencies=random_adjacencies,
        gold_edges=gold_edges,
        min_qwen_recall=min_qwen_recall,
    )
    write_quality_report(report, output_path)
    return report


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _load_gold_specs(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("gold_edges", [])
    if not isinstance(raw, list):
        raise ValueError("gold spec file must be a JSON list or object with gold_edges")
    return [dict(item) for item in raw]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score v24 graph-prior implanted-signal recovery against matched random.")
    parser.add_argument("--source-weight-graph-dir", required=True)
    parser.add_argument("--gold-specs", required=True, help="JSON file with gold block specs.")
    parser.add_argument("--output", required=True, help="Path for graph_prior_quality_report.json.")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--random-seeds", default="0,1,2,3,4")
    parser.add_argument("--graph-scope", default="attention_mlp_moe", choices=["mlp_only", "attention_mlp", "attention_mlp_moe", "all"])
    parser.add_argument("--edge-score-name", default="normalized_frobenius")
    parser.add_argument("--min-qwen-recall", type=float, default=0.95)
    args = parser.parse_args()

    report = run_prior_recovery_quality(
        source_weight_graph_dir=args.source_weight_graph_dir,
        gold_block_specs=_load_gold_specs(args.gold_specs),
        output_path=args.output,
        k=args.k,
        random_seeds=_parse_int_list(args.random_seeds),
        graph_scope=args.graph_scope,
        edge_score_name=args.edge_score_name,
        min_qwen_recall=args.min_qwen_recall,
    )
    if not math.isfinite(float(report["delta_recovery"])):
        raise SystemExit("invalid delta_recovery")
    print(f"wrote {args.output}")
    print(f"quality_ok={report['quality_ok']} delta_recovery={report['delta_recovery']:.6f}")


if __name__ == "__main__":
    main()
