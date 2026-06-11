from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .qwen_weight_graph import (
    WeightGraphEdge,
    WeightGraphResult,
    load_weight_graph_as_world_graph,
    read_weight_graph_artifacts,
)


SCHEMA_VERSION = "qwen_graph_prior_eval.v1"
DEFAULT_K_VALUES = [4, 8, 16]
DEFAULT_BASELINES = ["dense_full", "hand_k4", "learned_k4", "random_matched"]
UNAVAILABLE_METRIC = None


@dataclass(frozen=True)
class PriorEdge:
    edge_id: str
    src_id: str
    dst_id: str
    relation: str
    weight: float
    score_name: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "src_id": self.src_id,
            "dst_id": self.dst_id,
            "relation": self.relation,
            "weight": self.weight,
            "score_name": self.score_name,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class PriorAdjacency:
    name: str
    source: str
    k: int | None
    edges: tuple[PriorEdge, ...]
    node_count: int

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def as_summary(self) -> dict[str, Any]:
        relation_counts = Counter(edge.relation for edge in self.edges)
        score_counts = Counter(edge.score_name for edge in self.edges)
        return {
            "adjacency_name": self.name,
            "source": self.source,
            "k": self.k,
            "edge_count": self.edge_count,
            "node_count": self.node_count,
            "relation_counts": dict(sorted(relation_counts.items())),
            "score_name_counts": dict(sorted(score_counts.items())),
        }


def _hash_id(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:16]


def _node_kind_by_id(result: WeightGraphResult) -> dict[str, str]:
    return {node.node_id: node.node_type for node in result.nodes}


def _node_features_by_id(result: WeightGraphResult) -> dict[str, dict[str, Any]]:
    return {node.node_id: dict(node.features) for node in result.nodes}


def _node_allowed(node_kind: str, features: dict[str, Any], graph_scope: str) -> bool:
    if graph_scope in {"all", "attention_mlp_moe"}:
        return True
    if graph_scope == "mlp_only":
        if node_kind in {"mlp_block", "expert", "router"}:
            return True
        if node_kind == "projection":
            return features.get("module") == "mlp" or bool(features.get("is_shared_expert"))
        return False
    if graph_scope == "attention_mlp":
        return node_kind not in {"expert", "router"}
    raise ValueError(f"unknown graph_scope={graph_scope!r}")


def _edge_allowed(
    edge: WeightGraphEdge,
    *,
    kinds: dict[str, str],
    features: dict[str, dict[str, Any]],
    graph_scope: str,
    edge_score_name: str,
) -> bool:
    if edge_score_name != "any" and edge.score_name != edge_score_name:
        return False
    src_kind = kinds.get(edge.src_id, "")
    dst_kind = kinds.get(edge.dst_id, "")
    return (
        _node_allowed(src_kind, features.get(edge.src_id, {}), graph_scope)
        and _node_allowed(dst_kind, features.get(edge.dst_id, {}), graph_scope)
    )


def _to_prior_edge(edge: WeightGraphEdge, *, source: str) -> PriorEdge:
    metadata = {
        "source_tensor": edge.source_tensor,
        "provenance": edge.provenance,
    }
    return PriorEdge(
        edge_id=edge.edge_id,
        src_id=edge.src_id,
        dst_id=edge.dst_id,
        relation=edge.rel,
        weight=float(edge.weight),
        score_name=edge.score_name,
        source=source,
        metadata=metadata,
    )


def build_qwen_topk_adjacency(
    result: WeightGraphResult,
    *,
    k: int,
    graph_scope: str = "attention_mlp_moe",
    edge_score_name: str = "normalized_frobenius",
) -> PriorAdjacency:
    """Build A_qwen = per-source TopK(ScoreEdges(G_0), K) from graph artifacts only."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    kinds = _node_kind_by_id(result)
    features = _node_features_by_id(result)
    by_src: dict[str, list[WeightGraphEdge]] = defaultdict(list)
    for edge in result.edges:
        if _edge_allowed(edge, kinds=kinds, features=features, graph_scope=graph_scope, edge_score_name=edge_score_name):
            by_src[edge.src_id].append(edge)

    selected: list[WeightGraphEdge] = []
    for src_id in sorted(by_src):
        edges = sorted(
            by_src[src_id],
            key=lambda e: (-float(e.weight), e.rel, e.dst_id, e.edge_id),
        )
        selected.extend(edges[:k])

    prior_edges = tuple(_to_prior_edge(edge, source="G_0") for edge in selected)
    node_ids = {edge.src_id for edge in prior_edges} | {edge.dst_id for edge in prior_edges}
    return PriorAdjacency(
        name=f"qwen_topk_k{k}",
        source="G_0",
        k=k,
        edges=prior_edges,
        node_count=len(node_ids),
    )


def _degree_weights(values: Iterable[str]) -> dict[str, int]:
    counts = Counter(values)
    return {str(k): int(v) for k, v in counts.items()}


def _weighted_choice(rng: random.Random, ids: list[str], weights: dict[str, int]) -> str:
    if not ids:
        raise ValueError("cannot sample from an empty node bucket")
    total = sum(max(1, int(weights.get(node_id, 1))) for node_id in ids)
    pick = rng.uniform(0.0, float(total))
    acc = 0.0
    for node_id in ids:
        acc += max(1, int(weights.get(node_id, 1)))
        if pick <= acc:
            return node_id
    return ids[-1]


def build_matched_random_adjacency(
    qwen: PriorAdjacency,
    result: WeightGraphResult,
    *,
    seed: int,
) -> PriorAdjacency:
    """
    Build A_random with matched edge count and coarse relation/node-type distribution.

    The sampler also uses the qwen prior's in/out degree frequencies as sampling
    weights. It emits synthetic edge ids and never copies raw tensor data.
    """
    rng = random.Random(seed)
    kinds = _node_kind_by_id(result)
    nodes_by_kind: dict[str, list[str]] = defaultdict(list)
    for node_id, kind in sorted(kinds.items()):
        nodes_by_kind[kind].append(node_id)

    out_weights = _degree_weights(edge.src_id for edge in qwen.edges)
    in_weights = _degree_weights(edge.dst_id for edge in qwen.edges)

    random_edges: list[PriorEdge] = []
    used_ids: set[str] = set()
    for ordinal, edge in enumerate(qwen.edges):
        src_kind = kinds.get(edge.src_id, "")
        dst_kind = kinds.get(edge.dst_id, "")
        src_bucket = nodes_by_kind.get(src_kind, [])
        dst_bucket = nodes_by_kind.get(dst_kind, [])
        if not src_bucket or not dst_bucket:
            raise ValueError(f"missing node bucket for matched random edge {ordinal}")

        src_id = _weighted_choice(rng, src_bucket, out_weights)
        dst_id = _weighted_choice(rng, dst_bucket, in_weights)
        edge_id = _hash_id("random_matched", str(seed), str(ordinal), src_id, dst_id, edge.relation)
        salt = 0
        while edge_id in used_ids:
            salt += 1
            edge_id = _hash_id("random_matched", str(seed), str(ordinal), str(salt), src_id, dst_id, edge.relation)
        used_ids.add(edge_id)
        random_edges.append(
            PriorEdge(
                edge_id=edge_id,
                src_id=src_id,
                dst_id=dst_id,
                relation=edge.relation,
                weight=edge.weight,
                score_name=edge.score_name,
                source=f"random_matched_seed_{seed}",
                metadata={
                    "matched_qwen_edge_id": edge.edge_id,
                    "matched_src_node_kind": src_kind,
                    "matched_dst_node_kind": dst_kind,
                    "random_seed": seed,
                    "ordinal": ordinal,
                },
            )
        )

    node_ids = {edge.src_id for edge in random_edges} | {edge.dst_id for edge in random_edges}
    return PriorAdjacency(
        name=f"random_matched_k{qwen.k}_seed{seed}",
        source="random_matched",
        k=qwen.k,
        edges=tuple(random_edges),
        node_count=len(node_ids),
    )


def _unavailable_baseline_row(name: str, source: str, k: int | None = None) -> dict[str, Any]:
    return {
        "adjacency_name": name,
        "source": source,
        "k": k,
        "edge_count": UNAVAILABLE_METRIC,
        "node_count": UNAVAILABLE_METRIC,
        "route_acc": UNAVAILABLE_METRIC,
        "generic_acc": UNAVAILABLE_METRIC,
        "affine_acc": UNAVAILABLE_METRIC,
        "memory_mb": UNAVAILABLE_METRIC,
        "block_ms_median": UNAVAILABLE_METRIC,
        "quality_ok": False,
        "memory_ok": False,
        "speed_ok": False,
        "metrics_available": False,
    }


def _adjacency_matrix_row(
    adjacency: PriorAdjacency,
    *,
    quality_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fields = dict(quality_fields or {})
    return {
        "adjacency_name": adjacency.name,
        "source": adjacency.source,
        "k": adjacency.k,
        "edge_count": adjacency.edge_count,
        "node_count": adjacency.node_count,
        "route_acc": UNAVAILABLE_METRIC,
        "generic_acc": UNAVAILABLE_METRIC,
        "affine_acc": UNAVAILABLE_METRIC,
        "memory_mb": UNAVAILABLE_METRIC,
        "block_ms_median": UNAVAILABLE_METRIC,
        "quality_ok": bool(fields.get("graph_prior_quality_ok", False)),
        "memory_ok": False,
        "speed_ok": False,
        "metrics_available": bool(fields),
        **fields,
    }


def build_baseline_matrix(
    qwen_adjacencies: list[PriorAdjacency],
    random_adjacencies: list[PriorAdjacency],
    *,
    quality_fields_by_name: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    quality_fields_by_name = quality_fields_by_name or {}
    rows = [
        _unavailable_baseline_row("dense_full", "dense", None),
        _unavailable_baseline_row("hand_topology_k4", "hand", 4),
        _unavailable_baseline_row("learned_topology_k4", "champion", 4),
    ]
    rows.extend(
        _adjacency_matrix_row(adj, quality_fields=quality_fields_by_name.get(adj.name))
        for adj in random_adjacencies
    )
    rows.extend(
        _adjacency_matrix_row(adj, quality_fields=quality_fields_by_name.get(adj.name))
        for adj in qwen_adjacencies
    )
    return rows


def _manifest_hash(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_gold_specs(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("gold_edges", [])
    if not isinstance(raw, list):
        raise ValueError("gold spec file must be a JSON list or object with gold_edges")
    return [dict(item) for item in raw]


def _quality_fields_from_report(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {
        str(report["adjacency_name"]): {
            "graph_prior_recall_at_k": report["qwen_recall_at_k"],
            "graph_prior_precision_at_k": report["qwen_precision_at_k"],
            "graph_prior_hit_count": report["qwen_hit_count"],
            "graph_prior_pred_count": report["qwen_pred_count"],
            "graph_prior_gold_edge_count": report["gold_edge_count"],
            "graph_prior_delta_recovery": report["delta_recovery"],
            "graph_prior_random_recall_mean": report["random_recall_mean"],
            "graph_prior_random_recall_std": report["random_recall_std"],
            "graph_prior_quality_ok": bool(report["quality_ok"]),
            "graph_prior_metric": "implanted_signal_recovery",
        }
    }
    for row in report.get("random_rows", []) or []:
        out[str(row["adjacency_name"])] = {
            "graph_prior_recall_at_k": row["recall_at_k"],
            "graph_prior_precision_at_k": row["precision_at_k"],
            "graph_prior_hit_count": row["hit_count"],
            "graph_prior_pred_count": row["pred_count"],
            "graph_prior_gold_edge_count": report["gold_edge_count"],
            "graph_prior_delta_recovery": None,
            "graph_prior_random_recall_mean": None,
            "graph_prior_random_recall_std": None,
            "graph_prior_quality_ok": False,
            "graph_prior_metric": "implanted_signal_recovery",
        }
    return out


def _build_graph_prior_quality_report(
    *,
    result: WeightGraphResult,
    qwen_adjacencies: list[PriorAdjacency],
    random_adjacencies: list[PriorAdjacency],
    gold_block_specs: list[dict[str, Any]],
    edge_score_name: str,
    min_qwen_recall: float,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    from .qwen_graph_prior_quality import (
        SCHEMA_VERSION as QUALITY_SCHEMA_VERSION,
        evaluate_prior_recovery,
        gold_edges_from_block_specs,
    )

    gold_edges = gold_edges_from_block_specs(result, gold_block_specs, score_name=edge_score_name)
    reports = []
    fields_by_name: dict[str, dict[str, Any]] = {}
    for qwen_adj in qwen_adjacencies:
        matched_random = [adj for adj in random_adjacencies if adj.k == qwen_adj.k]
        report = evaluate_prior_recovery(
            qwen_adjacency=qwen_adj,
            random_adjacencies=matched_random,
            gold_edges=gold_edges,
            min_qwen_recall=min_qwen_recall,
        )
        reports.append(report)
        fields_by_name.update(_quality_fields_from_report(report))

    primary = reports[0] if reports else {
        "qwen_recall_at_k": 0.0,
        "qwen_precision_at_k": 0.0,
        "random_recall_mean": 0.0,
        "random_recall_std": 0.0,
        "delta_recovery": 0.0,
        "quality_ok": False,
    }
    aggregate = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "status": "implanted_signal_recovery",
        "qwen_recall_at_k": primary["qwen_recall_at_k"],
        "qwen_precision_at_k": primary["qwen_precision_at_k"],
        "random_recall_mean": primary["random_recall_mean"],
        "random_recall_std": primary["random_recall_std"],
        "delta_recovery": primary["delta_recovery"],
        "quality_ok": bool(primary["quality_ok"]),
        "primary_adjacency_name": primary.get("adjacency_name"),
        "gold_edge_count": primary.get("gold_edge_count", len(gold_edges)),
        "rows": reports,
    }
    return aggregate, fields_by_name


def run_graph_prior_eval(
    *,
    source_weight_graph_dir: str | Path,
    output_dir: str | Path,
    k_values: list[int] | None = None,
    random_seeds: list[int] | None = None,
    graph_scope: str = "attention_mlp_moe",
    edge_score_name: str = "normalized_frobenius",
    quality_dataset: str = "",
    runtime_protocol: str = "unavailable",
    memory_protocol: str = "unavailable",
    gold_block_specs: list[dict[str, Any]] | None = None,
    min_qwen_recall: float = 0.95,
) -> dict[str, Any]:
    source_dir = Path(source_weight_graph_dir)
    out_dir = Path(output_dir)
    ks = list(k_values or DEFAULT_K_VALUES)
    seeds = list(random_seeds or [0])

    result = read_weight_graph_artifacts(source_dir)
    world = load_weight_graph_as_world_graph(source_dir)
    manifest_dict = result.manifest.as_dict()

    qwen_adjacencies = [
        build_qwen_topk_adjacency(
            result,
            k=k,
            graph_scope=graph_scope,
            edge_score_name=edge_score_name,
        )
        for k in ks
    ]
    random_adjacencies = [
        build_matched_random_adjacency(qwen_adj, result, seed=seed)
        for qwen_adj in qwen_adjacencies
        for seed in seeds
    ]

    graph_prior_quality_report: dict[str, Any] | None = None
    quality_fields_by_name: dict[str, dict[str, Any]] = {}
    if gold_block_specs is not None:
        graph_prior_quality_report, quality_fields_by_name = _build_graph_prior_quality_report(
            result=result,
            qwen_adjacencies=qwen_adjacencies,
            random_adjacencies=random_adjacencies,
            gold_block_specs=gold_block_specs,
            edge_score_name=edge_score_name,
            min_qwen_recall=min_qwen_recall,
        )

    baseline_matrix = build_baseline_matrix(
        qwen_adjacencies,
        random_adjacencies,
        quality_fields_by_name=quality_fields_by_name,
    )

    prior_config = {
        "schema_version": SCHEMA_VERSION,
        "source_weight_graph_dir": str(source_dir),
        "source_manifest_hash": _manifest_hash(manifest_dict),
        "graph_scope": graph_scope,
        "block_size": result.manifest.block_size,
        "topk": ks,
        "edge_score_name": edge_score_name,
        "selection_policy": "per_source_topk_score_desc",
        "random_seed": seeds,
        "baseline_set": DEFAULT_BASELINES + [f"qwen_topk_k{k}" for k in ks],
        "quality_dataset": quality_dataset,
        "runtime_protocol": runtime_protocol,
        "memory_protocol": memory_protocol,
        "graph_prior_quality_protocol": "implanted_signal_recovery" if gold_block_specs is not None else "unavailable",
        "min_qwen_recall": min_qwen_recall,
        "teacher_checkpoint_loaded": False,
        "champion_scorer_mutated": False,
    }
    adjacency_summary = {
        "schema_version": SCHEMA_VERSION,
        "world_node_count": world.node_count(),
        "world_edge_count": world.edge_count(),
        "qwen": [adj.as_summary() for adj in qwen_adjacencies],
        "random_matched": [adj.as_summary() for adj in random_adjacencies],
    }
    quality_report = {
        "schema_version": SCHEMA_VERSION,
        "status": "graph_prior_quality_run" if graph_prior_quality_report is not None else "quality_not_run",
        "promotable": False,
        "graph_prior_quality_ok": None if graph_prior_quality_report is None else bool(graph_prior_quality_report["quality_ok"]),
        "reason": (
            "implanted-signal recovery only; sparse-student task quality is still unavailable"
            if graph_prior_quality_report is not None
            else "v24 P0 graph-prior loader does not run sparse-student quality yet"
        ),
    }
    memory_report = {
        "schema_version": SCHEMA_VERSION,
        "status": "memory_not_measured",
        "memory_ok": False,
    }
    runtime_report = {
        "schema_version": SCHEMA_VERSION,
        "status": "runtime_not_measured",
        "speed_ok": False,
        "repeated_speed_distribution_ok": False,
    }
    promotion_decision = {
        "schema_version": SCHEMA_VERSION,
        "decision": "speed_pending",
        "promote": False,
        "quality_ok": False,
        "memory_ok": False,
        "speed_ok": False,
        "old_champion_scorer_behavior_unchanged": True,
        "reason": "baseline matrix has topology artifacts only; quality/runtime/memory gates unavailable",
    }

    _write_json(out_dir / "prior_config.json", prior_config)
    _write_json(out_dir / "baseline_matrix.json", baseline_matrix)
    _write_csv(out_dir / "baseline_matrix.csv", baseline_matrix)
    _write_json(out_dir / "adjacency_summary.json", adjacency_summary)
    _write_json(out_dir / "quality_report.json", quality_report)
    if graph_prior_quality_report is not None:
        _write_json(out_dir / "graph_prior_quality_report.json", graph_prior_quality_report)
    _write_json(out_dir / "memory_report.json", memory_report)
    _write_json(out_dir / "runtime_report.json", runtime_report)
    _write_jsonl(out_dir / "paired_regression_report.jsonl", [])
    _write_json(out_dir / "promotion_decision.json", promotion_decision)

    return {
        "prior_config": prior_config,
        "baseline_matrix": baseline_matrix,
        "adjacency_summary": adjacency_summary,
        "quality_report": quality_report,
        "graph_prior_quality_report": graph_prior_quality_report,
        "memory_report": memory_report,
        "runtime_report": runtime_report,
        "promotion_decision": promotion_decision,
    }


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v24 Qwen graph-prior adjacency and matched-random baseline artifacts.")
    parser.add_argument("--source-weight-graph-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--random-seeds", default="0,1,2,3,4")
    parser.add_argument("--graph-scope", default="attention_mlp_moe", choices=["mlp_only", "attention_mlp", "attention_mlp_moe", "all"])
    parser.add_argument("--edge-score-name", default="normalized_frobenius")
    parser.add_argument("--quality-dataset", default="")
    parser.add_argument("--runtime-protocol", default="unavailable")
    parser.add_argument("--memory-protocol", default="unavailable")
    parser.add_argument("--gold-specs", default="", help="Optional JSON gold block specs for implanted-signal recovery.")
    parser.add_argument("--min-qwen-recall", type=float, default=0.95)
    args = parser.parse_args()

    gold_specs = _load_gold_specs(args.gold_specs) if args.gold_specs else None
    result = run_graph_prior_eval(
        source_weight_graph_dir=args.source_weight_graph_dir,
        output_dir=args.output_dir,
        k_values=_parse_int_list(args.k_values),
        random_seeds=_parse_int_list(args.random_seeds),
        graph_scope=args.graph_scope,
        edge_score_name=args.edge_score_name,
        quality_dataset=args.quality_dataset,
        runtime_protocol=args.runtime_protocol,
        memory_protocol=args.memory_protocol,
        gold_block_specs=gold_specs,
        min_qwen_recall=args.min_qwen_recall,
    )
    print(f"wrote {args.output_dir}")
    print(f"rows={len(result['baseline_matrix'])}")


if __name__ == "__main__":
    main()
