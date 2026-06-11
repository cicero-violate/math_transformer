from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .embedder import MathEmbedder
from .eval import _load_route_eval_records
from .learned_topology import topk_mask_from_scores
from .learned_topology_runtime import LearnedTopologyBuilder
from .model import MathRoutedTransformer
from .normalize import normalize
from .parser import parse
from .router import EXPERT_NAMES
from .topology import TopologyBuilder, build_hand_score_matrix
from .topology_trace import hash_nodes, summarize_mask, summarize_overlap


def _load_dense_model(checkpoint: str | None, device: torch.device, **cfg: Any):
    model = MathRoutedTransformer(
        **cfg,
        dropout=0.0,
        attention_mode="full",
        share_topology_cache=False,
    ).to(device)
    if checkpoint:
        try:
            state = torch.load(checkpoint, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
    model.eval()
    return model, model.state_dict()


def _make_sparse_model(dense_state, device: torch.device, *, learned_scorer_checkpoint: str | None = None, **cfg: Any):
    fixed_k = int(cfg["fixed_k"])
    model = MathRoutedTransformer(
        **cfg,
        dropout=0.0,
        attention_mode="neighbor_sparse",
        max_neighbors=fixed_k,
        share_topology_cache=True,
        sparse_selector="topology_only",
    ).to(device)
    model.load_state_dict(dense_state)
    if learned_scorer_checkpoint:
        for layer in model.layers:
            layer.topology = LearnedTopologyBuilder(
                learned_scorer_checkpoint,
                fixed_k=fixed_k,
                topk=cfg["topk"],
                local_window=cfg["local_window"],
                middle_bridge_width=cfg["middle_bridge_width"],
                device=device,
            )
            layer.max_neighbors = fixed_k
    model.eval()
    return model


def _pred_for(model, nodes, env, device: torch.device, pass_nodes: bool) -> int:
    with torch.no_grad():
        x = model.embed_nodes(nodes).to(device)
        out = model(x, nodes if pass_nodes else None, env=env)[0]
        logits = model.route_logits(out)
        return int(logits.argmax(dim=-1).item())


def _edge_list(mask: np.ndarray, nodes, scores: np.ndarray | None = None) -> list[dict[str, Any]]:
    out = []
    rows, cols = np.nonzero(mask)
    for i, j in zip(rows.tolist(), cols.tolist()):
        item = {
            "src": i,
            "dst": j,
            "src_label": repr(nodes[i]),
            "dst_label": repr(nodes[j]),
        }
        if scores is not None:
            item["score"] = float(scores[i, j])
        out.append(item)
    return out


def _edge_delta(mask_a: np.ndarray, mask_b: np.ndarray, nodes, scores: np.ndarray | None = None):
    return _edge_list(mask_a & ~mask_b, nodes, scores=scores)


def _pred_name(pred: int):
    return EXPERT_NAMES[pred] if 0 <= pred < len(EXPERT_NAMES) else pred


def export_topology_edge_deltas(
    *,
    examples_path: str,
    checkpoint: str,
    learned_scorer_checkpoint: str,
    output: str | Path = "runs/diagnostics/learned_k4_vs_hand_k4_edge_deltas.jsonl",
    k: int = 4,
    device: str | None = "auto",
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 128,
    topk: int = 3,
    local_window: int = 1,
    topology_mode: str = "middle_preserving_topk",
    middle_bridge_width: int = 0,
    max_records: int | None = None,
) -> int:
    if device in (None, "auto"):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    torch.manual_seed(0)
    cfg = dict(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        topk=topk,
        local_window=local_window,
        topology_mode=topology_mode,
        fixed_k=k,
        middle_bridge_width=middle_bridge_width,
    )
    dense, dense_state = _load_dense_model(checkpoint, dev, **cfg)
    hand = _make_sparse_model(dense_state, dev, **cfg)
    learned = _make_sparse_model(
        dense_state,
        dev,
        learned_scorer_checkpoint=learned_scorer_checkpoint,
        **cfg,
    )
    embedder = MathEmbedder()
    hand_builder = TopologyBuilder(
        topk=topk,
        local_window=local_window,
        topology_mode=topology_mode,
        fixed_k=k,
        middle_bridge_width=middle_bridge_width,
    )
    learned_builder = LearnedTopologyBuilder(
        learned_scorer_checkpoint,
        fixed_k=k,
        topk=topk,
        local_window=local_window,
        middle_bridge_width=middle_bridge_width,
        device=dev,
    )
    records = _load_route_eval_records(examples_path)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for sample_id, rec in enumerate(records):
            root = normalize(parse(rec["expr"]))
            nodes = root.collect_nodes()
            env = rec["env"] or None
            target = int(rec["expert_id"])
            dense_pred = _pred_for(dense, nodes, env, dev, pass_nodes=False)
            hand_pred = _pred_for(hand, nodes, env, dev, pass_nodes=True)
            learned_pred = _pred_for(learned, nodes, env, dev, pass_nodes=True)
            hand_correct = hand_pred == target
            learned_correct = learned_pred == target
            if hand_correct == learned_correct:
                continue

            z = embedder.encode_batch(nodes)
            hand_mask, _ = hand_builder.build_scored_topk(nodes, z, env)
            hand_scores, _ = build_hand_score_matrix(
                nodes,
                z,
                env,
                local_window=local_window,
                include_middle_bridge=(topology_mode == "middle_preserving_topk"),
                middle_bridge_width=middle_bridge_width,
            )
            learned_scores_t = learned_builder._scores(nodes, z, env, dev)
            learned_mask_t = topk_mask_from_scores(learned_scores_t, k)
            learned_scores = learned_scores_t.detach().float().cpu().numpy()
            learned_mask = learned_mask_t.detach().bool().cpu().numpy()
            union_i, union_j = np.nonzero(hand_mask | learned_mask)
            rec_out = {
                "sample_id": sample_id,
                "expr": rec.get("expr"),
                "nodes_hash": hash_nodes(nodes),
                "target_expert": rec.get("expert"),
                "target_expert_id": target,
                "dense_pred": _pred_name(dense_pred),
                "dense_pred_id": dense_pred,
                "hand_pred": _pred_name(hand_pred),
                "hand_pred_id": hand_pred,
                "learned_pred": _pred_name(learned_pred),
                "learned_pred_id": learned_pred,
                "dense_correct": dense_pred == target,
                "hand_correct": hand_correct,
                "learned_correct": learned_correct,
                "outcome": "learned_win" if learned_correct and not hand_correct else "learned_loss",
                "k": k,
                "node_labels": [repr(node) for node in nodes],
                "target_edges": _edge_list(hand_mask, nodes, scores=hand_scores),
                "hand_edges": _edge_list(hand_mask, nodes, scores=hand_scores),
                "learned_edges": _edge_list(learned_mask, nodes, scores=learned_scores),
                "removed_edges": _edge_delta(hand_mask, learned_mask, nodes, scores=hand_scores),
                "extra_edges": _edge_delta(learned_mask, hand_mask, nodes, scores=learned_scores),
                "edge_scores": {
                    "hand": {f"{i}->{j}": float(hand_scores[i, j]) for i, j in zip(union_i.tolist(), union_j.tolist())},
                    "learned": {f"{i}->{j}": float(learned_scores[i, j]) for i, j in zip(union_i.tolist(), union_j.tolist())},
                },
                "summaries": {
                    "hand_mask": summarize_mask(torch.as_tensor(hand_mask, dtype=torch.bool)),
                    "learned_mask": summarize_mask(torch.as_tensor(learned_mask, dtype=torch.bool)),
                    "overlap": summarize_overlap(
                        torch.as_tensor(learned_mask, dtype=torch.bool),
                        torch.as_tensor(hand_mask, dtype=torch.bool),
                    ),
                },
            }
            fh.write(json.dumps(rec_out, sort_keys=True) + "\n")
            n_written += 1
            if max_records is not None and n_written >= max_records:
                break
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export edge-level deltas for learned-vs-hand paired quality flips.")
    parser.add_argument("--examples", default="data/synthetic_hard/val.jsonl", dest="examples_path")
    parser.add_argument("--checkpoint", default="runs/checkpoints/synthetic_hard_dense.pt")
    parser.add_argument("--learned-scorer-checkpoint", default="runs/checkpoints/topology_scorer.champion.pt", dest="learned_scorer_checkpoint")
    parser.add_argument("--output", default="runs/diagnostics/learned_k4_vs_hand_k4_edge_deltas.jsonl")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--quality-device", default="auto", dest="device")
    parser.add_argument("--d-model", type=int, default=64, dest="d_model")
    parser.add_argument("--n-heads", type=int, default=4, dest="n_heads")
    parser.add_argument("--n-layers", type=int, default=2, dest="n_layers")
    parser.add_argument("--d-ff", type=int, default=128, dest="d_ff")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--local-window", type=int, default=1, dest="local_window")
    parser.add_argument("--topology-mode", default="middle_preserving_topk", choices=["scored_topk", "middle_preserving_topk"])
    parser.add_argument("--middle-bridge-width", type=int, default=0, dest="middle_bridge_width")
    parser.add_argument("--max-records", type=int, default=None, dest="max_records")
    args = parser.parse_args()
    count = export_topology_edge_deltas(**vars(args))
    print(f"wrote {args.output}")
    print(f"records={count}")


if __name__ == "__main__":
    main()
