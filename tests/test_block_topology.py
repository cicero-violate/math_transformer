from __future__ import annotations

import torch

from src.block_learned_topology import HeuristicBlockTopologyBuilder
from src.ir import add, matmul, var
from src.topology_cache import TopologyCache


def _nodes(n: int):
    expr = add(matmul(var("A"), var("x")), var("b"))
    base = expr.collect_nodes()
    return [base[i % len(base)] for i in range(n)]


def test_block_topology_shapes_small_cpu():
    nodes = _nodes(130)
    builder = HeuristicBlockTopologyBuilder(
        block_size=64,
        topk_blocks=4,
        block_token_cap=8,
        fixed_k=16,
        device="cpu",
    )
    prepared = builder.prepare_topology(nodes, device=torch.device("cpu"), max_neighbors=16)
    assert prepared.block_neighbors.shape[0] == 3
    assert prepared.token_neighbors.shape[0] == 130
    assert prepared.token_valid_i8.shape == prepared.token_neighbors.shape
    assert prepared.token_neighbors.shape[1] == 16
    assert prepared.diagnostics.by_relation["block_score_entries"] == 9


def test_block_topology_prepared_cache_n4096_cpu():
    nodes = _nodes(4096)
    builder = HeuristicBlockTopologyBuilder(
        block_size=64,
        topk_blocks=4,
        block_token_cap=4,
        fixed_k=16,
        device="cpu",
    )
    cache = TopologyCache()
    prepared = cache.get_or_prepare(nodes, None, None, builder, max_neighbors=16, device=torch.device("cpu"))
    assert prepared.neighbors.shape == (4096, 16)
    assert prepared.valid_i8.shape == (4096, 16)
    assert prepared.block_neighbors is not None
    assert prepared.block_valid_i8 is not None
    assert prepared.block_neighbors.shape == (64, 7)
    assert prepared.block_valid_i8.shape == prepared.block_neighbors.shape
    assert prepared.block_size == 64
    assert prepared.is_block_topology is True
    assert prepared.diagnostics.by_relation["block_count"] == 64
    assert prepared.diagnostics.by_relation["block_score_entries"] == 4096
    assert builder.last_timing["topology_prepare_ms"] >= 0.0
    assert builder.last_timing["neighbor_table_build_ms"] >= 0.0
