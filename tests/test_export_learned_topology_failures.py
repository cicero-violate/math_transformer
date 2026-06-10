from __future__ import annotations

import torch

from src.export_learned_topology_failures import build_failure_trace_record
from src.normalize import normalize
from src.parser import parse


def _nodes(expr: str):
    return normalize(parse(expr)).collect_nodes()


def test_build_failure_trace_record_uses_common_trace_schema():
    nodes = _nodes("add(x, y)")
    hand_mask = torch.tensor([
        [True, True, False],
        [False, True, True],
        [False, False, True],
    ])
    learned_mask = torch.tensor([
        [True, False, True],
        [False, True, False],
        [False, True, True],
    ])
    scores = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    rec = {"expr": "add(x, y)", "expert": "generic_expert", "expert_id": 1, "env": {"x": [4], "y": [4]}}

    row = build_failure_trace_record(
        sample_id=7,
        rec=rec,
        nodes=nodes,
        hand_mask=hand_mask,
        learned_mask=learned_mask,
        learned_scores=scores,
        scorer_checkpoint="runs/checkpoints/scorer.pt",
        hand_k=3,
        learned_k=3,
        dense_pred=0,
        hand_pred=1,
        learned_pred=2,
        hidden_l1=0.1,
        hidden_cos=0.9,
        logit_l1=0.2,
        logit_kl=0.03,
    )

    assert row["sample_id"] == 7
    assert row["domain"] == "math"
    assert row["feature_schema"] == "topology_edge_features.v1"
    assert row["env"] == {"x": [4], "y": [4]}
    assert row["scores"]["numel"] == 9
    assert row["target_topology"]["active_edges"] == 5
    assert row["pred_topology"]["active_edges"] == 5
    assert row["overlap"]["edge_hits"] == 3
    assert row["prediction"]["target_expert"] == "generic_expert"
    assert row["prediction"]["learned_correct"] is False
    assert row["agreement"]["hidden_cos"] == 0.9
    assert row["diagnostics"]["trace_source"] == "export_learned_topology_failures"
    assert row["diagnostics"]["failure"] is True
    assert row["diagnostics"]["failure_type"] == "route_miss"
    assert row["diagnostics"]["missing_edge_count"] == 2
    assert row["diagnostics"]["extra_edge_count"] == 2
