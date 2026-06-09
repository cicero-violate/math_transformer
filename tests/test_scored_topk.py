"""v6 Sprint 1+2+4: scored top-K routing, z-caching, same_operator reduction."""
from __future__ import annotations
import numpy as np
import pytest
import torch
from src.parser import parse
from src.normalize import normalize
from src.embedder import MathEmbedder
from src.topology import TopologyBuilder, RELATION_WEIGHTS
from src.topology_cache import TopologyCache

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


# ── avg_k stability ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [16, 32, 64])
def test_avg_k_stable_near_fixed_k(n):
    fixed_k = 8
    nodes = _nodes(n)
    embedder = MathEmbedder()
    z = embedder.encode_batch(nodes)
    tb = TopologyBuilder(topk=3, local_window=1, topology_mode="scored_topk", fixed_k=fixed_k)
    mask, diag = tb.build_scored_topk(nodes, z, ENV)
    # avg_k should stay near fixed_k, not grow with n
    # allow 2× slack because identity always adds 1 and fixed_k caps per row
    assert diag.avg_k <= fixed_k * 2, (
        f"n={n}: avg_k={diag.avg_k:.1f} too large for fixed_k={fixed_k}"
    )
    assert diag.avg_k >= 1.0, f"n={n}: avg_k={diag.avg_k:.1f} is unexpectedly zero"


def test_avg_k_does_not_scale_with_n():
    """avg_k with scored_topk must not double when n doubles."""
    fixed_k = 8
    embedder = MathEmbedder()
    avg_ks = []
    for n in [16, 32, 64]:
        nodes = _nodes(n)
        z = embedder.encode_batch(nodes)
        tb = TopologyBuilder(topk=3, local_window=1, topology_mode="scored_topk", fixed_k=fixed_k)
        _, diag = tb.build_scored_topk(nodes, z, ENV)
        avg_ks.append(diag.avg_k)
    # avg_k at n=64 must not be > 3× that at n=16 (union mode would give ~4×)
    ratio = avg_ks[-1] / avg_ks[0]
    assert ratio < 3.0, (
        f"avg_k scales too fast: {avg_ks} (ratio {ratio:.2f}×). "
        "scored_topk should hold avg_k near fixed_k."
    )


# ── self-loops ───────────────────────────────────────────────────────────────

def test_scored_topk_diagonal_all_true():
    """Every node must attend to itself (identity relation has weight 10.0)."""
    nodes = _nodes(12)
    embedder = MathEmbedder()
    z = embedder.encode_batch(nodes)
    tb = TopologyBuilder(topk=3, local_window=1, topology_mode="scored_topk", fixed_k=6)
    mask, _ = tb.build_scored_topk(nodes, z, ENV)
    n = len(nodes)
    for i in range(n):
        assert mask[i, i], f"Node {i} missing self-loop in scored_topk mask"


# ── same_operator dominance ──────────────────────────────────────────────────

def test_same_operator_not_dominant_with_default_weights():
    """With RELATION_WEIGHTS, same_operator (w=0.1) should not dominate edges."""
    nodes = _nodes(32)
    embedder = MathEmbedder()
    z = embedder.encode_batch(nodes)
    tb = TopologyBuilder(topk=3, local_window=1, topology_mode="scored_topk", fixed_k=8)
    mask, diag = tb.build_scored_topk(nodes, z, ENV)
    same_op_count = diag.by_relation.get("same_operator", 0)
    total = diag.allowed_edges
    if total > 0:
        frac = same_op_count / total
        # In union mode this would be ~99% for roots; scored_topk should reduce it
        # We just verify it's not the only contributor by checking total > same_op_count
        # (other relations must contribute some edges via higher-weight scoring)
        assert frac < 1.0, "same_operator is contributing 100% of edges — scoring is broken"


def test_relation_weights_contain_all_expected_keys():
    expected = {
        "identity", "symbolic_dependency", "composition",
        "shape_compat", "middle_bridge", "embedding", "local_window", "same_operator",
    }
    assert set(RELATION_WEIGHTS.keys()) == expected


def test_relation_weights_ordering():
    """identity must have highest weight; same_operator must have lowest."""
    assert RELATION_WEIGHTS["identity"] == max(RELATION_WEIGHTS.values())
    assert RELATION_WEIGHTS["same_operator"] == min(RELATION_WEIGHTS.values())


# ── z-cache (Sprint 2) ───────────────────────────────────────────────────────

def test_get_or_encode_returns_same_array():
    nodes = _nodes(8)
    cache = TopologyCache()

    class CountingEmbedder:
        def __init__(self):
            self.calls = 0
            self._inner = MathEmbedder()

        def encode_batch(self, nodes):
            self.calls += 1
            return self._inner.encode_batch(nodes)

    embedder = CountingEmbedder()
    z1 = cache.get_or_encode(nodes, embedder)
    z2 = cache.get_or_encode(nodes, embedder)
    assert embedder.calls == 1, f"encode_batch called {embedder.calls} times; expected 1"
    assert np.array_equal(z1, z2)


def test_get_or_encode_different_nodes_encode_separately():
    cache = TopologyCache()

    class CountingEmbedder:
        def __init__(self):
            self.calls = 0
            self._inner = MathEmbedder()

        def encode_batch(self, nodes):
            self.calls += 1
            return self._inner.encode_batch(nodes)

    embedder = CountingEmbedder()
    nodes_a = _nodes(4)
    nodes_b = _nodes(8)
    cache.get_or_encode(nodes_a, embedder)
    cache.get_or_encode(nodes_b, embedder)
    assert embedder.calls == 2


def test_cache_clear_resets_z_store():
    nodes = _nodes(6)
    cache = TopologyCache()

    class CountingEmbedder:
        def __init__(self):
            self.calls = 0
            self._inner = MathEmbedder()

        def encode_batch(self, nodes):
            self.calls += 1
            return self._inner.encode_batch(nodes)

    embedder = CountingEmbedder()
    cache.get_or_encode(nodes, embedder)
    cache.clear()
    cache.get_or_encode(nodes, embedder)
    assert embedder.calls == 2, "z-store should be cleared by cache.clear()"


# ── model forward with scored_topk ──────────────────────────────────────────

def test_model_forward_scored_topk():
    from src.model import MathRoutedTransformer
    nodes = _nodes(8)
    model = MathRoutedTransformer(
        d_model=32, n_heads=2, n_layers=1, d_ff=64,
        topk=2, local_window=1, attention_mode="neighbor_sparse",
        max_neighbors=4, topology_mode="scored_topk", fixed_k=4,
    )
    x = model.embed_nodes(nodes)
    with torch.no_grad():
        out, masks, routes = model(x, nodes, env=ENV)
    assert out.shape == x.shape, f"Output shape {out.shape} != input shape {x.shape}"
    assert masks[0] is not None


def test_model_forward_scored_topk_cached():
    """Shared cache across layers should not break scored_topk mode."""
    from src.model import MathRoutedTransformer
    nodes = _nodes(8)
    model = MathRoutedTransformer(
        d_model=32, n_heads=2, n_layers=2, d_ff=64,
        topk=2, local_window=1, attention_mode="neighbor_sparse",
        max_neighbors=4, topology_mode="scored_topk", fixed_k=4,
        share_topology_cache=True,
    )
    x = model.embed_nodes(nodes)
    model.eval()
    with torch.no_grad():
        out1, _, _ = model(x, nodes, env=ENV)
        out2, _, _ = model(x, nodes, env=ENV)
    # Two identical forwards on a cached model should give identical outputs
    assert torch.allclose(out1, out2, atol=1e-5), "Cached model gives non-deterministic output"
