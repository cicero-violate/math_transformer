from __future__ import annotations

import torch

from src.embedder import MathEmbedder
from src.learned_topology import (
    FEATURE_NAMES,
    LearnedTopologyScorer,
    build_edge_feature_tensor,
    learned_topk_mask,
    topk_mask_from_scores,
)
from src.normalize import normalize
from src.parser import parse


def _nodes(expr: str):
    return normalize(parse(expr)).collect_nodes()


def test_edge_feature_tensor_shape_and_identity_feature():
    nodes = _nodes("add(matmul(A, x), b)")
    z = MathEmbedder().encode_batch(nodes)
    features = build_edge_feature_tensor(
        nodes,
        z,
        {"A": (32, 64), "x": (64,), "b": (32,)},
        local_window=1,
        middle_bridge_width=1,
    )

    assert features.shape == (len(nodes), len(nodes), len(FEATURE_NAMES))
    assert torch.all(features.diagonal(dim1=0, dim2=1)[0] == 1.0)


def test_learned_scorer_outputs_square_score_matrix():
    nodes = _nodes("constraint(leq(matmul(A, x), b))")
    features = build_edge_feature_tensor(nodes)
    scorer = LearnedTopologyScorer(feature_dim=features.shape[-1], hidden_dim=16)
    scores = scorer(features)

    assert scores.shape == (len(nodes), len(nodes))


def test_topk_mask_from_scores_forces_self_and_caps_rows():
    scores = torch.randn(10, 10)
    mask = topk_mask_from_scores(scores, fixed_k=4)

    assert mask.shape == (10, 10)
    assert torch.all(torch.diag(mask))
    assert int(mask.sum(dim=1).max().item()) <= 4


def test_learned_topk_mask_smoke():
    nodes = _nodes("matmul(A, add(x, y))")
    scorer = LearnedTopologyScorer(hidden_dim=16)
    mask, scores = learned_topk_mask(
        scorer,
        nodes,
        env={"A": (8, 8), "x": (8,), "y": (8,)},
        fixed_k=3,
    )

    assert mask.shape == scores.shape == (len(nodes), len(nodes))
    assert torch.all(torch.diag(mask))
    assert int(mask.sum(dim=1).max().item()) <= 3