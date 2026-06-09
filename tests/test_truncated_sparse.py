"""Sprint 5 — fixed-K truncated sparse safety tests."""
import numpy as np
import torch
import pytest
from src.ir import add, matmul, var
from src.parser import parse
from src.normalize import normalize
from src.embedder import MathEmbedder
from src.topology import TopologyBuilder, build_priority_matrix
from src.sparse_attention import neighbors_from_mask_prioritized, max_k_from_mask
from src.model import MathRoutedTransformerBlock, MathRoutedTransformer


def _mask_and_priority(nodes, topk=3, local_window=1, env=None):
    z = MathEmbedder().encode_batch(nodes)
    tb = TopologyBuilder(topk=topk, local_window=local_window)
    np_mask, diag = tb.build_detailed(nodes, z, env)
    mask_t = torch.tensor(np_mask, dtype=torch.bool)
    priority = build_priority_matrix(nodes, z=z, env=env, topk=topk, local_window=local_window)
    return mask_t, priority, diag


def _sample_nodes(n=8):
    exprs = ["add(matmul(A, x), b)", "matmul(Q, K)", "grad(f, x)", "sum(i, x_i)"]
    roots = [normalize(parse(e)) for e in exprs]
    nodes = []
    while len(nodes) < n:
        nodes.extend(roots[:n - len(nodes)])
    return nodes[:n]


# ── Safety: no row should have zero valid neighbors ───────────────────────────

def test_truncated_no_empty_rows():
    """Every row must have at least 1 valid neighbor (self-attention guaranteed)."""
    nodes = _sample_nodes(8)
    mask_t, priority, diag = _mask_and_priority(nodes)

    # Use max_k=1 (extreme truncation)
    nb, valid = neighbors_from_mask_prioritized(mask_t, priority, max_k=1)
    per_row = valid.sum(dim=-1)
    assert (per_row >= 1).all(), f"Rows with 0 valid: {(per_row == 0).nonzero()}"


# ── Priority: identity edges survive truncation ───────────────────────────────

def test_identity_edges_retained_under_truncation():
    """Self edges (priority=1) must appear in every row's truncated list."""
    nodes = _sample_nodes(8)
    mask_t, priority, _ = _mask_and_priority(nodes)

    nb, valid = neighbors_from_mask_prioritized(mask_t, priority, max_k=1)
    T = mask_t.shape[0]
    for i in range(T):
        valid_nb = nb[i, valid[i]].tolist()
        assert i in valid_nb, f"Row {i}: self-index missing from max_k=1 result: {valid_nb}"


# ── Priority: symbolic before same_operator under truncation ─────────────────

def test_symbolic_dep_before_same_operator_under_truncation():
    """When truncating, symbolic dep edges (priority 2) beat same_op (priority 7)."""
    # Construct a tree where root has both symbolic dep and same_op candidates
    a = var("a")
    b = var("b")
    root = add(a, b)
    # root → a, root → b: symbolic dep
    # a → b: same_operator (both var)
    # Use max_k=2: for root, neighbors should include self (priority 1) + a or b (priority 2)
    # before any same_op edges
    nodes = [root, a, b]
    mask_t, priority, _ = _mask_and_priority(nodes, topk=0, local_window=0)

    # root row (i=0) should pick a/b before any same_op pair
    nb, valid = neighbors_from_mask_prioritized(mask_t, priority, max_k=2)
    row0_nb = nb[0, valid[0]].tolist()
    # Expected: self (0) + one of (1 or 2) for symbolic dep
    symbolic_present = any(j in row0_nb for j in [1, 2])
    assert symbolic_present, f"Row 0: no symbolic dep neighbor in max_k=2 result: {row0_nb}"


# ── Model with max_neighbors ──────────────────────────────────────────────────

def test_model_max_neighbors_output_shape():
    T, D = 8, 64
    nodes = _sample_nodes(T)
    model = MathRoutedTransformer(
        d_model=D, n_heads=4, n_layers=1, d_ff=128,
        topk=2, local_window=1, attention_mode="neighbor_sparse",
        max_neighbors=3,
    )
    x = model.embed_nodes(nodes)
    with torch.no_grad():
        out, masks, routes = model(x, nodes)
    assert out.shape == (1, T, D)


def test_truncated_k_less_than_max_k():
    """Truncated neighbor list must use K ≤ max_k_from_mask."""
    nodes = _sample_nodes(8)
    mask_t, priority, diag = _mask_and_priority(nodes)
    max_k = diag.max_k
    trunc_k = max(1, max_k // 2)

    nb_trunc, valid_trunc = neighbors_from_mask_prioritized(mask_t, priority, trunc_k)
    # No row's valid count exceeds trunc_k
    per_row = valid_trunc.sum(dim=-1)
    assert (per_row <= trunc_k).all(), f"Some rows have > {trunc_k} valid neighbors"


def test_exact_sparse_matches_dense_masked():
    """exact_sparse (K=max_k) must match dense masked attention output."""
    from src.attention import math_attention
    from src.sparse_attention import neighbor_attention

    torch.manual_seed(3)
    B, H, T, D = 1, 2, 8, 16

    nodes = _sample_nodes(T)
    mask_t, priority, diag = _mask_and_priority(nodes)
    K = diag.max_k
    nb, valid = neighbors_from_mask_prioritized(mask_t, priority, K)

    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    out_dense = math_attention(q, k, v, mask_t.unsqueeze(0).unsqueeze(0))
    out_sparse = neighbor_attention(q, k, v, nb, valid)

    assert torch.allclose(out_dense, out_sparse, atol=1e-5), (
        f"Max diff: {(out_dense - out_sparse).abs().max().item():.2e}"
    )
