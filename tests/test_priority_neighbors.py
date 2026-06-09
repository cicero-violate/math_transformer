"""Sprint 2 — priority parity: model and benchmark must produce identical neighbors."""
import numpy as np
import torch
import pytest
from src.parser import parse
from src.normalize import normalize
from src.ir import MathNode
from src.embedder import MathEmbedder
from src.topology import TopologyBuilder, build_priority_matrix
from src.topology_cache import TopologyCache
from src.sparse_attention import neighbors_from_mask_prioritized, max_k_from_mask
from src.model import MathRoutedTransformerBlock

ENV = {"A": (32, 64), "x": (64,), "b": (32,)}
EXPRS = ["add(matmul(A, x), b)", "matmul(A, x)", "grad(f, x)", "sum(i, x_i)"]


def _nodes(n: int = 8) -> list[MathNode]:
    roots = [normalize(parse(e)) for e in EXPRS]
    nodes = []
    while len(nodes) < n:
        nodes.extend(roots[:n - len(nodes)])
    return nodes[:n]


def test_priority_matrix_shape():
    nodes = _nodes(6)
    z = MathEmbedder().encode_batch(nodes)
    P = build_priority_matrix(nodes, z=z, env=ENV, topk=2, local_window=1)
    assert P.shape == (6, 6)
    assert P.dtype == np.int8


def test_priority_matrix_identity_is_lowest_value():
    """Identity/self edges must have the lowest (highest-priority) value."""
    nodes = _nodes(6)
    z = MathEmbedder().encode_batch(nodes)
    P = build_priority_matrix(nodes, z=z, env=ENV, topk=2, local_window=1)
    diag_vals = P[np.arange(6), np.arange(6)]
    # All diagonal values should be the minimum priority value (=1 for identity)
    assert (diag_vals == 1).all(), f"Diagonal has non-1 values: {diag_vals}"


def test_model_and_benchmark_same_neighbors():
    """
    Given the same nodes/env/topk/local_window, model block and standalone
    priority-based neighbor selection must produce identical neighbors.
    """
    n, D = 8, 64
    nodes = _nodes(n)
    embedder = MathEmbedder()
    z = embedder.encode_batch(nodes)

    tb = TopologyBuilder(topk=2, local_window=1)
    np_mask, diag = tb.build_detailed(nodes, z, ENV)
    mask_t = torch.tensor(np_mask, dtype=torch.bool)
    priority = build_priority_matrix(nodes, z=z, env=ENV, topk=2, local_window=1)

    # Benchmark-side neighbors (exact = max_k)
    K = diag.max_k
    nb_bench, valid_bench = neighbors_from_mask_prioritized(mask_t, priority, K)

    # Model-side: use cached topology path
    cache = TopologyCache()
    block = MathRoutedTransformerBlock(
        d_model=D, n_heads=4, d_ff=128,
        topk=2, local_window=1, attention_mode="neighbor_sparse",
        topology_cache=cache,
    )
    import torch.nn as nn
    proj = nn.Linear(z.shape[1], D, bias=False)
    x = proj(torch.from_numpy(z).float()).unsqueeze(0)
    with torch.no_grad():
        block(x, nodes, env=ENV)

    cached = cache.get_or_build(nodes, z, ENV, tb, max_neighbors=None)
    nb_model, valid_model = cached.neighbors, cached.valid

    assert torch.equal(nb_model, nb_bench), "Neighbor indices differ"
    assert torch.equal(valid_model, valid_bench), "Valid tensors differ"


def test_symbolic_edges_before_same_operator_edges():
    """
    For two nodes that share an operator AND have a symbolic dependency,
    the symbolic dependency priority (2) must beat same_operator (7).
    Uses topk=0 and local_window=0 to isolate symbolic_dep vs same_operator.
    """
    from src.ir import add, var
    a = var("a")
    b = var("b")
    root = add(a, b)  # root depends on a and b; a and b share "var" op
    nodes = [root, a, b]
    z = MathEmbedder().encode_batch(nodes)
    # topk=0: disable embedding_topk so only sym_dep/same_op/identity remain
    P = build_priority_matrix(nodes, z=z, env=None, topk=0, local_window=0)
    # root(0) → a(1): must be symbolic dep (priority 2)
    assert P[0, 1] == 2, \
        f"Expected root→a priority=2 (symbolic_dep), got {P[0, 1]}"
    # a(1) → b(2): same_operator only (both var), no symbolic dep → priority 7
    assert P[1, 2] == 7, \
        f"Expected a→b priority=7 (same_op), got {P[1, 2]}"
    # Verify symbolic outranks same_op numerically (lower = higher priority)
    assert P[0, 1] < P[1, 2], \
        "symbolic_dep priority must be lower (higher priority) than same_operator"
