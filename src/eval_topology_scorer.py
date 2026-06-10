from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .eval import _load_route_eval_records
from .embedder import MathEmbedder
from .learned_topology import (
    FEATURE_NAMES,
    LearnedTopologyScorer,
    build_edge_feature_tensor,
    topk_mask_from_scores,
)
from .normalize import normalize
from .parser import parse
from .topology import TopologyBuilder, build_hand_score_matrix
from .topology_trace import (
    TopologyTraceWriter,
    hash_nodes,
    summarize_mask,
    summarize_overlap,
    summarize_scores,
)
from .train_topology_scorer import _topk_mask_from_teacher_scores


def _resolve_device(device_name: str | None) -> torch.device:
    if device_name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return dev


def _load_scorer(checkpoint: str, device: torch.device) -> tuple[LearnedTopologyScorer, dict]:
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    hidden_dim = int(state.get("hidden_dim", 64))
    feature_names = tuple(state.get("feature_names", FEATURE_NAMES))
    scorer = LearnedTopologyScorer(feature_dim=len(feature_names), hidden_dim=hidden_dim).to(device)
    scorer.load_state_dict(state["model_state_dict"])
    scorer.eval()
    return scorer, state


def evaluate_topology_scorer(
    *,
    examples_path: str,
    checkpoint: str,
    eval_k: int = 8,
    target_k: int = 16,
    local_window: int = 1,
    middle_bridge_width: int = 1,
    topology_mode: str = "middle_preserving_topk",
    device: str | None = "auto",
    max_examples: int = 0,
    trace_output: str | None = None,
) -> dict[str, float | int | str]:
    if not Path(examples_path).exists():
        raise FileNotFoundError(f"examples file not found: {examples_path}")
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    dev = _resolve_device(device)
    scorer, ckpt_state = _load_scorer(checkpoint, dev)
    feature_names = tuple(ckpt_state.get("feature_names", FEATURE_NAMES))
    records = _load_route_eval_records(examples_path)
    if max_examples and max_examples > 0:
        records = records[:max_examples]
    if not records:
        raise ValueError(f"no route examples loaded from {examples_path}")

    embedder = MathEmbedder()
    # Keep the teacher builder construction as an explicit record of the target
    # topology family even though target masks are produced from hand scores below.
    TopologyBuilder(
        topk=1,
        local_window=local_window,
        topology_mode=topology_mode,
        fixed_k=target_k,
        middle_bridge_width=middle_bridge_width,
    )
    hits_total = 0.0
    pred_total = 0.0
    target_total = 0.0
    row_recall_sum = 0.0
    row_precision_sum = 0.0
    row_count = 0
    self_hits = 0
    self_total = 0
    row_cap_violations = 0
    graph_count = 0
    n_min = 10**9
    n_max = 0
    trace_writer = TopologyTraceWriter(trace_output) if trace_output else None

    try:
        with torch.no_grad():
            for sample_idx, rec in enumerate(records):
                root = normalize(parse(rec["expr"]))
                nodes = root.collect_nodes()
                z = embedder.encode_batch(nodes)
                env = rec["env"] or None
                teacher_score_np, _ = build_hand_score_matrix(
                    nodes,
                    z,
                    env,
                    local_window=local_window,
                    include_middle_bridge=(topology_mode == "middle_preserving_topk"),
                    middle_bridge_width=middle_bridge_width,
                )
                teacher_score = torch.tensor(teacher_score_np, dtype=torch.float32, device=dev)
                target = _topk_mask_from_teacher_scores(teacher_score, eval_k)
                features = build_edge_feature_tensor(
                    nodes,
                    z,
                    env,
                    local_window=local_window,
                    middle_bridge_width=middle_bridge_width,
                    device=dev,
                )
                scores = scorer(features)
                pred = topk_mask_from_scores(scores, eval_k)
                hits = (pred & target).sum(dim=1).float()
                pred_rows = pred.sum(dim=1).clamp(min=1).float()
                target_rows = target.sum(dim=1).clamp(min=1).float()
                row_recall_sum += float((hits / target_rows).sum().item())
                row_precision_sum += float((hits / pred_rows).sum().item())
                row_count += int(pred.shape[0])
                hits_total += float(hits.sum().item())
                pred_total += float(pred.sum().item())
                target_total += float(target.sum().item())
                per_sample_row_cap_violations = 0
                if pred.numel():
                    self_hits += int(torch.diag(pred).sum().item())
                    self_total += int(pred.shape[0])
                    per_sample_row_cap_violations = int((pred.sum(dim=1) > eval_k).sum().item())
                    row_cap_violations += per_sample_row_cap_violations

                if trace_writer is not None:
                    trace_writer.write({
                        "sample_id": sample_idx,
                        "domain": "math",
                        "expr": rec.get("expr"),
                        "nodes_hash": hash_nodes(nodes),
                        "n": len(nodes),
                        "k": eval_k,
                        "scorer_checkpoint": checkpoint,
                        "feature_schema": "topology_edge_features.v1",
                        "feature_names": list(feature_names),
                        "topology_config": {
                            "eval_k": eval_k,
                            "target_k": target_k,
                            "local_window": local_window,
                            "middle_bridge_width": middle_bridge_width,
                            "topology_mode": topology_mode,
                        },
                        "features": {
                            "shape": list(features.shape),
                            "dtype": str(features.dtype).replace("torch.", ""),
                        },
                        "scores": summarize_scores(scores),
                        "pred_topology": summarize_mask(pred),
                        "target_topology": summarize_mask(target),
                        "overlap": summarize_overlap(pred, target),
                        "target": {
                            "expert": rec.get("expert"),
                            "expert_id": rec.get("expert_id"),
                        },
                        "diagnostics": {
                            "row_cap_violation_count": per_sample_row_cap_violations,
                            "self_loop_count": int(torch.diag(pred).sum().item()) if pred.numel() else 0,
                        },
                    })
                graph_count += 1
                n_min = min(n_min, len(nodes))
                n_max = max(n_max, len(nodes))
    finally:
        if trace_writer is not None:
            trace_writer.close()

    summary: dict[str, float | int | str] = {
        "examples": graph_count,
        "device": str(dev),
        "checkpoint": checkpoint,
        "eval_k": eval_k,
        "target_k": target_k,
        "trained_target_k": int(ckpt_state.get("trained_target_k", ckpt_state.get("eval_k", eval_k))),
        "mean_row_recall": row_recall_sum / max(row_count, 1),
        "mean_row_precision": row_precision_sum / max(row_count, 1),
        "micro_recall": hits_total / max(target_total, 1.0),
        "micro_precision": hits_total / max(pred_total, 1.0),
        "self_loop_rate": self_hits / max(self_total, 1),
        "row_cap_violations": row_cap_violations,
        "n_min": 0 if n_min == 10**9 else n_min,
        "n_max": n_max,
    }
    if trace_output:
        summary["trace_output"] = trace_output
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a learned topology scorer against compressed hand-topology targets.")
    parser.add_argument("--examples", default="data/synthetic_hard/val.jsonl")
    parser.add_argument("--checkpoint", default="runs/checkpoints/topology_scorer.champion.pt")
    parser.add_argument("--eval-k", type=int, default=8)
    parser.add_argument("--target-k", type=int, default=16)
    parser.add_argument("--local-window", type=int, default=1)
    parser.add_argument("--middle-bridge-width", type=int, default=1)
    parser.add_argument("--topology-mode", default="middle_preserving_topk", choices=["scored_topk", "middle_preserving_topk"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--trace-output", default=None, help="Optional JSONL path for compact topology traces.")
    args = parser.parse_args()
    summary = evaluate_topology_scorer(
        examples_path=args.examples,
        checkpoint=args.checkpoint,
        eval_k=args.eval_k,
        target_k=args.target_k,
        local_window=args.local_window,
        middle_bridge_width=args.middle_bridge_width,
        topology_mode=args.topology_mode,
        device=args.device,
        max_examples=args.max_examples,
        trace_output=args.trace_output,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
