from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .eval import _load_route_eval_records
from .learned_topology import FEATURE_NAMES, topk_mask_from_scores
from .learned_topology_runtime import LearnedTopologyBuilder
from .model import MathRoutedTransformer
from .normalize import normalize
from .parser import parse
from .tasks import ID_TO_EXPERT
from .topology import TopologyBuilder
from .topology_trace import (
    TopologyTraceWriter,
    hash_nodes,
    summarize_mask,
    summarize_overlap,
    summarize_scores,
)


def _resolve_device(name: str) -> torch.device:
    if name in ("", "auto", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return dev


def _expert_name(idx: int) -> str | int:
    return ID_TO_EXPERT.get(idx, idx)


def _mask_tensor(mask: torch.Tensor | Any) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        return mask.detach().bool().cpu()
    return torch.as_tensor(mask, dtype=torch.bool)


def _edge_set(mask: torch.Tensor | Any) -> set[tuple[int, int]]:
    m = _mask_tensor(mask)
    return {(int(i), int(j)) for i, j in m.nonzero(as_tuple=False).tolist()}


def build_failure_trace_record(
    *,
    sample_id: int,
    rec: dict[str, Any],
    nodes: list[Any],
    hand_mask: torch.Tensor | Any,
    learned_mask: torch.Tensor | Any,
    learned_scores: torch.Tensor | None,
    scorer_checkpoint: str,
    hand_k: int,
    learned_k: int,
    dense_pred: int,
    hand_pred: int,
    learned_pred: int,
    hidden_l1: float,
    hidden_cos: float,
    logit_l1: float,
    logit_kl: float,
) -> dict[str, Any]:
    """Return one failure row using the common compact topology trace schema."""
    hand_t = _mask_tensor(hand_mask)
    learned_t = _mask_tensor(learned_mask)
    hand_edges = _edge_set(hand_t)
    learned_edges = _edge_set(learned_t)
    true_id = int(rec["expert_id"])
    is_generic = rec.get("expert") == "generic_expert"
    missing_edges = sorted(hand_edges - learned_edges)
    extra_edges = sorted(learned_edges - hand_edges)

    return {
        "sample_id": sample_id,
        "domain": "math",
        "expr": rec.get("expr"),
        "nodes_hash": hash_nodes(nodes),
        "n": len(nodes),
        "k": learned_k,
        "scorer_checkpoint": scorer_checkpoint,
        "feature_schema": "topology_edge_features.v1",
        "feature_names": list(FEATURE_NAMES),
        "topology_config": {
            "hand_k": hand_k,
            "learned_k": learned_k,
            "topk": 3,
            "local_window": 1,
            "middle_bridge_width": 1,
            "topology_mode": "learned_topology",
            "target_topology_mode": "middle_preserving_topk",
        },
        "scores": summarize_scores(learned_scores),
        "target_topology": summarize_mask(hand_t),
        "pred_topology": summarize_mask(learned_t),
        "overlap": summarize_overlap(learned_t, hand_t),
        "prediction": {
            "target_expert": rec.get("expert"),
            "target_expert_id": true_id,
            "dense_pred": _expert_name(dense_pred),
            "dense_pred_id": dense_pred,
            "dense_correct": dense_pred == true_id,
            "hand_pred": _expert_name(hand_pred),
            "hand_pred_id": hand_pred,
            "hand_correct": hand_pred == true_id,
            "learned_pred": _expert_name(learned_pred),
            "learned_pred_id": learned_pred,
            "learned_correct": learned_pred == true_id,
            "learned_dense_agree": learned_pred == dense_pred,
        },
        "agreement": {
            "hidden_l1": hidden_l1,
            "hidden_cos": hidden_cos,
            "logit_l1": logit_l1,
            "logit_kl": logit_kl,
        },
        "env": rec.get("env"),
        "diagnostics": {
            "trace_source": "export_learned_topology_failures",
            "failure": True,
            "failure_type": "route_miss",
            "is_generic_expert": is_generic,
            "missing_edges": missing_edges,
            "extra_edges": extra_edges,
            "missing_edge_count": len(missing_edges),
            "extra_edge_count": len(extra_edges),
            "learned_top_edges": sorted(learned_edges),
            "hand_top_edges": sorted(hand_edges),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export learned-topology route misses in the standard topology trace schema.")
    ap.add_argument("--examples", default="data/synthetic_hard/val.jsonl")
    ap.add_argument("--checkpoint", default="runs/checkpoints/synthetic_hard_dense.pt")
    ap.add_argument("--scorer", default="runs/checkpoints/topology_scorer.champion.pt")
    ap.add_argument("--hand-k", type=int, default=16)
    ap.add_argument("--learned-k", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output", default="runs/diagnostics/learned_topology_failures.jsonl")
    args = ap.parse_args()

    dev = _resolve_device(args.device)
    records = _load_route_eval_records(args.examples)
    base = dict(
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
        topk=3,
        local_window=1,
        topology_mode="middle_preserving_topk",
        fixed_k=args.hand_k,
        middle_bridge_width=1,
    )
    dense = MathRoutedTransformer(**base, attention_mode="full", share_topology_cache=False).to(dev)
    state = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    dense.load_state_dict(state)
    dense.eval()
    dense_state = dense.state_dict()

    hand = MathRoutedTransformer(
        **base,
        attention_mode="neighbor_sparse",
        max_neighbors=args.hand_k,
        share_topology_cache=True,
        sparse_selector="topology_only",
    ).to(dev)
    hand.load_state_dict(dense_state)
    hand.eval()

    learned = MathRoutedTransformer(
        **base,
        attention_mode="neighbor_sparse",
        max_neighbors=args.learned_k,
        share_topology_cache=True,
        sparse_selector="topology_only",
    ).to(dev)
    learned.load_state_dict(dense_state)
    for layer in learned.layers:
        layer.topology = LearnedTopologyBuilder(
            args.scorer,
            fixed_k=args.learned_k,
            topk=3,
            local_window=1,
            middle_bridge_width=1,
            device=dev,
        )
        layer.max_neighbors = args.learned_k
    learned.eval()

    hand_builder = TopologyBuilder(
        topk=3,
        local_window=1,
        topology_mode="middle_preserving_topk",
        fixed_k=args.hand_k,
        middle_bridge_width=1,
    )
    learned_builder = LearnedTopologyBuilder(
        args.scorer,
        fixed_k=args.learned_k,
        topk=3,
        local_window=1,
        middle_bridge_width=1,
        device=dev,
    )

    out = Path(args.output)
    generic_failures = 0
    failures_total = 0
    with TopologyTraceWriter(out) as writer, torch.no_grad():
        for idx, rec in enumerate(records):
            root = normalize(parse(rec["expr"]))
            nodes = root.collect_nodes()
            env = rec["env"] or None
            x = dense.embed_nodes(nodes).to(dev)

            dense_h = dense(x, None, env=env)[0]
            dense_logits = dense.route_logits(dense_h)
            dense_pred = int(dense_logits.argmax(dim=-1).item())

            hand_h = hand(x, nodes, env=env)[0]
            hand_logits = hand.route_logits(hand_h)
            hand_pred = int(hand_logits.argmax(dim=-1).item())

            learned_h = learned(x, nodes, env=env)[0]
            learned_logits = learned.route_logits(learned_h)
            learned_pred = int(learned_logits.argmax(dim=-1).item())

            true_id = int(rec["expert_id"])
            if learned_pred == true_id:
                continue

            hand_mask, _ = hand_builder.build_scored_topk(nodes, None, env)
            learned_scores = learned_builder._scores(nodes, None, env, dev)  # trace-only score summary
            learned_mask = topk_mask_from_scores(learned_scores, args.learned_k)

            hidden_l1 = float((learned_h.detach().float().cpu() - dense_h.detach().float().cpu()).abs().mean().item())
            hidden_cos = float(F.cosine_similarity(learned_h.reshape(1, -1), dense_h.reshape(1, -1), dim=1).item())
            logit_l1 = float((learned_logits.detach().float().cpu() - dense_logits.detach().float().cpu()).abs().mean().item())
            logit_kl = float(F.kl_div(F.log_softmax(learned_logits, dim=-1), F.softmax(dense_logits, dim=-1), reduction="batchmean").item())
            generic_failures += int(rec["expert"] == "generic_expert")
            failures_total += 1
            writer.write(build_failure_trace_record(
                sample_id=idx,
                rec=rec,
                nodes=nodes,
                hand_mask=hand_mask,
                learned_mask=learned_mask,
                learned_scores=learned_scores,
                scorer_checkpoint=args.scorer,
                hand_k=args.hand_k,
                learned_k=args.learned_k,
                dense_pred=dense_pred,
                hand_pred=hand_pred,
                learned_pred=learned_pred,
                hidden_l1=hidden_l1,
                hidden_cos=hidden_cos,
                logit_l1=logit_l1,
                logit_kl=logit_kl,
            ))

    print(f"wrote {out}")
    print(f"failures_total={failures_total}")
    print(f"generic_expert_failures={generic_failures}")


if __name__ == "__main__":
    main()
