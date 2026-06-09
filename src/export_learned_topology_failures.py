from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .eval import _load_route_eval_records
from .learned_topology_runtime import LearnedTopologyBuilder
from .model import MathRoutedTransformer
from .normalize import normalize
from .parser import parse
from .tasks import EXPERTS
from .topology import TopologyBuilder


def _resolve_device(name: str) -> torch.device:
    if name in ("", "auto", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return dev


def _expert_name(idx: int) -> str | int:
    return EXPERTS[idx] if 0 <= idx < len(EXPERTS) else idx


def _edge_set(mask) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j in zip(*mask.nonzero())}


def main() -> None:
    ap = argparse.ArgumentParser(description="Export learned-topology route misses.")
    ap.add_argument("--examples", default="data/synthetic_hard/val.jsonl")
    ap.add_argument("--checkpoint", default="runs/checkpoints/synthetic_hard_dense.pt")
    ap.add_argument("--scorer", default="runs/checkpoints/scorer_runtime_j_best.pt")
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

    failures: list[dict] = []
    generic_failures = 0
    with torch.no_grad():
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
            learned_mask, _ = learned_builder.build_scored_topk(nodes, None, env)
            hand_edges = _edge_set(hand_mask)
            learned_edges = _edge_set(learned_mask)

            hidden_l1 = float((learned_h.detach().float().cpu() - dense_h.detach().float().cpu()).abs().mean().item())
            hidden_cos = float(F.cosine_similarity(learned_h.reshape(1, -1), dense_h.reshape(1, -1), dim=1).item())
            logit_l1 = float((learned_logits.detach().float().cpu() - dense_logits.detach().float().cpu()).abs().mean().item())
            logit_kl = float(F.kl_div(F.log_softmax(learned_logits, dim=-1), F.softmax(dense_logits, dim=-1), reduction="batchmean").item())
            is_generic = rec["expert"] == "generic_expert"
            generic_failures += int(is_generic)
            failures.append({
                "example_id": idx,
                "expression": rec["expr"],
                "true_expert": rec["expert"],
                "dense_pred": _expert_name(dense_pred),
                "learned_pred": _expert_name(learned_pred),
                "hand_pred": _expert_name(hand_pred),
                "is_generic_expert": is_generic,
                "learned_top_edges": sorted(learned_edges),
                "hand_top_edges": sorted(hand_edges),
                "missing_edges": sorted(hand_edges - learned_edges),
                "extra_edges": sorted(learned_edges - hand_edges),
                "hidden_l1": hidden_l1,
                "hidden_cos": hidden_cos,
                "logit_l1": logit_l1,
                "logit_kl": logit_kl,
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in failures:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {out}")
    print(f"failures_total={len(failures)}")
    print(f"generic_expert_failures={generic_failures}")


if __name__ == "__main__":
    main()