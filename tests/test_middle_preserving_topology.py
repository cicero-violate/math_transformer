from __future__ import annotations

import numpy as np
import torch

from src.parser import parse
from src.normalize import normalize
from src.embedder import MathEmbedder
from src.topology import TopologyBuilder, build_priority_matrix
from src.middle_preserving_topology import (
    middle_anchor_indices,
    middle_bridge_matrix,
    middle_bridge_matrix_torch,
    middle_coverage_score,
)

EXPRS = [
    "add(matmul(A, x), b)",
    "matmul(A, x)",
    "grad(f, x)",
    "sum(i, x_i)",
    "matmul(Q, K)",
    "constraint(leq(matmul(A, x), b))",
]
ENV = {"A": (32, 64), "x": (64,), "b": (32,), "Q": (32, 64), "K": (64, 32)}


def _nodes(target_n: int):
    roots = [normalize(parse(e)) for e in EXPRS]
    nodes = []
    while len(nodes) < target_n:
        nodes.extend(roots[:target_n - len(nodes)])
    return nodes[:target_n]


def test_middle_anchor_indices_sorted_unique_and_bounded():
    anchors = middle_anchor_indices(16, width=1)
    assert anchors == sorted(set(anchors))
    assert all(0 <= i < 16 for i in anchors)
    assert 8 in anchors


def test_middle_bridge_matrix_shape_and_columns():
    mat = middle_bridge_matrix(12, width=0)
    assert mat.shape == (12, 12)
    assert mat.dtype == bool
    assert np.all(np.diag(mat))
    anchors = middle_anchor_indices(12)
    assert anchors
    for j in anchors:
        assert mat[:, j].all()


def test_middle_bridge_matrix_torch_matches_numpy():
    np_mat = middle_bridge_matrix(10, width=1)
    t_mat = middle_bridge_matrix_torch(10, width=1, device="cpu")
    assert torch.equal(t_mat.cpu(), torch.tensor(np_mat, dtype=torch.bool))


def test_middle_preserving_topk_increases_middle_coverage():
    nodes = _nodes(64)
    z = MathEmbedder().encode_batch(nodes)
    base = TopologyBuilder(
        topk=3, local_window=1, topology_mode="scored_topk", fixed_k=8,
    )
    v7 = TopologyBuilder(
        topk=3, local_window=1, topology_mode="middle_preserving_topk", fixed_k=8,
        middle_bridge_width=1,
    )
    base_mask, base_diag = base.build_scored_topk(nodes, z, ENV)
    v7_mask, v7_diag = v7.build_scored_topk(nodes, z, ENV)

    anchors = middle_anchor_indices(len(nodes), width=1)

    assert v7_diag.avg_k <= 8
    assert v7_diag.by_relation["middle_bridge"] > 0
    assert int(v7_mask[:, anchors].sum()) > int(base_mask[:, anchors].sum())


def test_middle_preserving_priority_contains_bridge_edges():
    nodes = _nodes(16)
    z = MathEmbedder().encode_batch(nodes)
    priority = build_priority_matrix(
        nodes, z=z, env=ENV, topk=1, local_window=1,
        include_middle_bridge=True, middle_bridge_width=0,
    )
    assert priority.shape == (16, 16)
    assert priority.dtype == np.int8
    anchors = middle_anchor_indices(16)
    assert anchors
    assert (priority[:, anchors] > 0).any()


def test_model_forward_middle_preserving_topk():
    from src.model import MathRoutedTransformer

    nodes = _nodes(12)
    model = MathRoutedTransformer(
        d_model=32, n_heads=2, n_layers=1, d_ff=64,
        topk=2, local_window=1, attention_mode="neighbor_sparse",
        max_neighbors=6, topology_mode="middle_preserving_topk", fixed_k=6,
        middle_bridge_width=1,
    )
    x = model.embed_nodes(nodes)
    model.eval()
    with torch.no_grad():
        out, masks, routes = model(x, nodes, env=ENV)
    assert out.shape == x.shape
    assert masks[0] is not None
