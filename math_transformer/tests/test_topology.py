import numpy as np
from src.ir import var, matmul, add
from src.parser import parse
from src.normalize import normalize
from src.topology import (
    TopologyBuilder, MaskDiagnostics,
    symbolic_dependency_matrix, same_operator_matrix,
    embedding_topk_matrix, local_window_matrix, identity_matrix,
)
from src.embedder import MathEmbedder


def _nodes(expr: str):
    return normalize(parse(expr)).collect_nodes()


def test_mask_shape():
    nodes = _nodes("add(matmul(A, x), b)")
    mask = TopologyBuilder().build(nodes)
    n = len(nodes)
    assert mask.shape == (n, n)
    assert mask.dtype == bool


def test_self_attention_always_true():
    nodes = _nodes("add(matmul(A, x), b)")
    mask = TopologyBuilder().build(nodes)
    assert np.all(np.diag(mask))


def test_parent_child_bidirectional():
    A, x = var("A"), var("x")
    root = matmul(A, x)
    nodes = root.collect_nodes()
    tb = TopologyBuilder(topk=0, local_window=0)
    mask = tb.build(nodes)
    root_idx = 0
    a_idx = nodes.index(A)
    x_idx = nodes.index(x)
    assert mask[root_idx, a_idx]
    assert mask[a_idx, root_idx]
    assert mask[root_idx, x_idx]
    assert mask[x_idx, root_idx]


def test_sparsity_below_one_for_large_sequence():
    exprs = ["add(matmul(A, x), b)", "matmul(Q, K)", "grad(f, x)", "sum(i, x_i)"] * 4
    nodes = [normalize(parse(e)) for e in exprs]
    tb = TopologyBuilder(topk=2, local_window=1)
    mask = tb.build(nodes)
    assert 0.0 < tb.sparsity_ratio(mask) < 1.0


def test_edge_count_positive():
    nodes = _nodes("matmul(A, x)")
    mask = TopologyBuilder(topk=1, local_window=1).build(nodes)
    assert TopologyBuilder().edge_count(mask) > 0


# ── Separate relation matrix tests (C1) ──────────────────────────────────────

def test_symbolic_dependency_matrix():
    A, x = var("A"), var("x")
    root = matmul(A, x)
    nodes = root.collect_nodes()
    mat = symbolic_dependency_matrix(nodes)
    root_idx, a_idx, x_idx = 0, nodes.index(A), nodes.index(x)
    assert mat[root_idx, a_idx]
    assert mat[a_idx, root_idx]
    assert not mat[a_idx, x_idx]  # siblings, not parent-child


def test_same_operator_matrix():
    nodes = _nodes("add(matmul(A, x), b)")
    mat = same_operator_matrix(nodes)
    n = len(nodes)
    assert mat.shape == (n, n)
    # No two nodes here share the same op (all unique in this expression)
    # Just check the matrix is symmetric and diagonal is False
    assert np.all(mat == mat.T)
    assert not np.any(np.diag(mat))


def test_local_window_matrix():
    mat = local_window_matrix(5, w=1)
    # Position 2 should attend to 1, 2, 3
    assert mat[2, 1] and mat[2, 2] and mat[2, 3]
    assert not mat[2, 0] and not mat[2, 4]


def test_identity_matrix():
    mat = identity_matrix(4)
    assert mat.shape == (4, 4)
    assert np.all(np.diag(mat))
    assert not np.any(mat & ~np.eye(4, dtype=bool))


def test_embedding_topk_matrix():
    nodes = [normalize(parse(e)) for e in
             ["add(matmul(A, x), b)", "add(matmul(W, h), c)", "grad(f, x)", "sum(i, x_i)"]]
    emb = MathEmbedder()
    Z = emb.encode_batch(nodes)
    mat = embedding_topk_matrix(Z, k=1)
    assert mat.shape == (4, 4)
    assert np.all(mat == mat.T)


# ── Diagnostics tests (C2) ────────────────────────────────────────────────────

def test_build_detailed_returns_diagnostics():
    nodes = _nodes("add(matmul(A, x), b)")
    tb = TopologyBuilder(topk=2, local_window=1)
    mask, diag = tb.build_detailed(nodes)
    assert isinstance(diag, MaskDiagnostics)
    assert diag.n == len(nodes)
    assert diag.full_edges == len(nodes) ** 2
    assert diag.allowed_edges == int(mask.sum())
    assert abs(diag.sparsity_ratio - diag.allowed_edges / diag.full_edges) < 1e-9
    assert abs(diag.relation_reduction - (1 - diag.sparsity_ratio)) < 1e-9


def test_diagnostics_by_relation_keys():
    nodes = _nodes("matmul(A, x)")
    _, diag = TopologyBuilder(topk=1, local_window=1).build_detailed(nodes)
    for key in ("symbolic_dependency", "same_operator", "embedding_topk",
                "local_window", "identity"):
        assert key in diag.by_relation


def test_tiny_config_not_saturated():
    # topk=1, local_window=1 should leave sparsity < 1.0 for sequences of ≥ 8 nodes
    exprs = ["add(matmul(A, x), b)", "grad(f, x)", "sum(i, x_i)", "matmul(Q, K)"] * 3
    nodes = [normalize(parse(e)) for e in exprs]
    tb = TopologyBuilder(topk=1, local_window=1)
    mask, diag = tb.build_detailed(nodes)
    assert diag.sparsity_ratio < 1.0, (
        f"Mask saturated with topk=1, local_window=1, n={len(nodes)}: {diag}"
    )
