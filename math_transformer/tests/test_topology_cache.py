"""Sprint 3 — topology cache correctness, hit/miss tracking, and speed."""
import numpy as np
import torch
import time
import pytest
from src.parser import parse
from src.normalize import normalize
from src.embedder import MathEmbedder
from src.topology import TopologyBuilder
from src.topology_cache import (
    TopologyCache,
    page_neighbor_table,
    stable_nodes_hash,
    stable_env_hash,
)

ENV = {"A": (32, 64), "x": (64,), "b": (32,)}
EXPRS = ["add(matmul(A, x), b)", "matmul(A, x)", "grad(f, x)", "sum(i, x_i)"]


def _nodes(n: int = 6):
    roots = [normalize(parse(e)) for e in EXPRS]
    nodes = []
    while len(nodes) < n:
        nodes.extend(roots[:n - len(nodes)])
    return nodes[:n]


def test_stable_nodes_hash_is_deterministic():
    nodes = _nodes(6)
    h1 = stable_nodes_hash(nodes)
    h2 = stable_nodes_hash(nodes)
    assert h1 == h2


def test_stable_nodes_hash_differs_for_different_nodes():
    n1 = _nodes(4)
    n2 = _nodes(6)
    assert stable_nodes_hash(n1) != stable_nodes_hash(n2)


def test_stable_env_hash_is_deterministic():
    h1 = stable_env_hash(ENV)
    h2 = stable_env_hash(ENV)
    assert h1 == h2


def test_stable_env_hash_none():
    assert stable_env_hash(None) == "no_env"


def test_cache_miss_on_first_call():
    cache = TopologyCache()
    nodes = _nodes(6)
    z = MathEmbedder().encode_batch(nodes)
    tb = TopologyBuilder(topk=2, local_window=1)
    cache.get_or_build(nodes, z, ENV, tb)
    assert cache.cache_misses == 1
    assert cache.cache_hits == 0


def test_cache_hit_on_second_call():
    cache = TopologyCache()
    nodes = _nodes(6)
    z = MathEmbedder().encode_batch(nodes)
    tb = TopologyBuilder(topk=2, local_window=1)
    cache.get_or_build(nodes, z, ENV, tb)
    cache.get_or_build(nodes, z, ENV, tb)
    assert cache.cache_hits == 1
    assert cache.cache_misses == 1


def test_cache_key_differs_for_different_env():
    cache = TopologyCache()
    nodes = _nodes(6)
    z = MathEmbedder().encode_batch(nodes)
    tb = TopologyBuilder(topk=2, local_window=1)
    env2 = {"A": (16, 32), "x": (32,)}
    cache.get_or_build(nodes, z, ENV, tb)
    cache.get_or_build(nodes, z, env2, tb)
    assert cache.cache_misses == 2


def test_cache_key_differs_for_different_topk():
    cache = TopologyCache()
    nodes = _nodes(6)
    z = MathEmbedder().encode_batch(nodes)
    tb1 = TopologyBuilder(topk=2, local_window=1)
    tb2 = TopologyBuilder(topk=3, local_window=1)
    cache.get_or_build(nodes, z, ENV, tb1)
    cache.get_or_build(nodes, z, ENV, tb2)
    assert cache.cache_misses == 2


def test_cached_mask_equals_uncached_mask():
    nodes = _nodes(6)
    z = MathEmbedder().encode_batch(nodes)
    tb = TopologyBuilder(topk=2, local_window=1)

    np_mask, _ = tb.build_detailed(nodes, z, ENV)
    uncached_mask = torch.tensor(np_mask, dtype=torch.bool)

    cache = TopologyCache()
    cached = cache.get_or_build(nodes, z, ENV, tb)

    assert torch.equal(cached.mask, uncached_mask)


def test_cached_diagnostics_equal_uncached():
    nodes = _nodes(6)
    z = MathEmbedder().encode_batch(nodes)
    tb = TopologyBuilder(topk=2, local_window=1)

    _, diag = tb.build_detailed(nodes, z, ENV)

    cache = TopologyCache()
    cached = cache.get_or_build(nodes, z, ENV, tb)

    assert cached.diagnostics.allowed_edges == diag.allowed_edges
    assert cached.diagnostics.by_relation == diag.by_relation


def test_cache_clear_resets_counts():
    cache = TopologyCache()
    nodes = _nodes(4)
    z = MathEmbedder().encode_batch(nodes)
    tb = TopologyBuilder(topk=1, local_window=1)
    cache.get_or_build(nodes, z, None, tb)
    cache.clear()
    assert cache.cache_hits == 0
    assert cache.cache_misses == 0
    assert len(cache) == 0


def test_cached_topology_build_faster_than_uncached():
    """
    Cache hit (topology already built) must be faster than cache miss
    (full build_detailed + priority_matrix + neighbor conversion).
    Measures the topology build directly, not the full block forward.
    """
    n = 32
    nodes = _nodes(n)
    z = MathEmbedder().encode_batch(nodes)
    tb = TopologyBuilder(topk=3, local_window=2)

    # Measure a cold build (cache miss path)
    n_cold = 20
    t_cold = 0.0
    for _ in range(n_cold):
        c = TopologyCache()
        t0 = time.perf_counter()
        c.get_or_build(nodes, z, ENV, tb)
        t_cold += time.perf_counter() - t0

    # Warm a shared cache once
    warm_cache = TopologyCache()
    warm_cache.get_or_build(nodes, z, ENV, tb)

    # Measure cache hit path
    n_hit = 100
    t_hit = 0.0
    for _ in range(n_hit):
        t0 = time.perf_counter()
        warm_cache.get_or_build(nodes, z, ENV, tb)
        t_hit += time.perf_counter() - t0

    avg_cold_ms = t_cold / n_cold * 1000
    avg_hit_ms  = t_hit  / n_hit  * 1000

    assert avg_hit_ms < avg_cold_ms, (
        f"Cache hit ({avg_hit_ms:.4f}ms) not faster than cold build ({avg_cold_ms:.4f}ms)"
    )


def test_page_neighbor_table_round_trips_materialization():
    neighbors = torch.arange(10 * 3).reshape(10, 3)
    valid = torch.ones(10, 3, dtype=torch.bool)
    valid[-1, -1] = False

    paged = page_neighbor_table(neighbors, valid, page_size=4)

    assert paged.num_pages == 3
    assert paged.padded_length == 12
    assert paged.neighbor_pages.shape == (3, 4, 3)
    assert torch.equal(paged.materialize_neighbors(), neighbors.long())
    assert torch.equal(paged.materialize_valid(), valid)
    assert torch.equal(paged.materialize_valid_i8(), valid.char())
    page_neighbors, page_valid_i8 = paged.page(0)
    assert page_neighbors.shape == (4, 3)
    assert page_valid_i8.dtype == torch.int8


def test_topology_cache_get_or_build_paged_uses_fixed_k_shape():
    nodes = _nodes(16)
    z = MathEmbedder().encode_batch(nodes)
    tb = TopologyBuilder(
        topk=2,
        local_window=1,
        topology_mode="middle_preserving_topk",
        fixed_k=4,
        middle_bridge_width=1,
    )
    cache = TopologyCache()
    paged_cached = cache.get_or_build_paged(
        nodes, z, ENV, tb,
        max_neighbors=4,
        page_size=5,
    )
    paged = paged_cached.paged_neighbors

    assert paged.length == 16
    assert paged.k == 4
    assert paged.page_size == 5
    assert paged.num_pages == 4
    assert paged.materialize_neighbors().shape == (16, 4)
    assert paged.materialize_valid_i8().shape == (16, 4)

    again = cache.get_or_build_paged(nodes, z, ENV, tb, max_neighbors=4, page_size=5)
    assert again is paged_cached
    assert cache.cache_hits >= 1
