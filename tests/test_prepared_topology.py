from __future__ import annotations

import torch

from src.embedder import MathEmbedder
from src.model import MathRoutedTransformer, MathRoutedTransformerBlock
from src.normalize import normalize
from src.parser import parse
from src.topology import TopologyBuilder
from src.topology_cache import TopologyCache

ENV = {"A": (32, 64), "x": (64,), "b": (32,)}
EXPRS = ["add(matmul(A, x), b)", "matmul(A, x)", "grad(f, x)", "sum(i, x_i)"]


def _nodes(n: int):
    roots = [normalize(parse(e)) for e in EXPRS]
    nodes = []
    while len(nodes) < n:
        nodes.extend(roots[: n - len(nodes)])
    return nodes[:n]


def test_get_or_prepare_middle_preserving_fixed_k():
    nodes = _nodes(16)
    z = MathEmbedder().encode_batch(nodes)
    tb = TopologyBuilder(topk=2, local_window=1, topology_mode="middle_preserving_topk", fixed_k=4, middle_bridge_width=1)
    cache = TopologyCache()
    prepared = cache.get_or_prepare(nodes, z, ENV, tb, max_neighbors=4, device=torch.device("cpu"))

    assert prepared.neighbors.shape == (16, 4)
    assert prepared.valid_i8.shape == (16, 4)
    assert prepared.valid_i8.dtype == torch.int8
    assert prepared.diagnostics.max_k == 4
    assert prepared.diagnostics.by_relation["middle_bridge"] > 0
    assert prepared.memory_bytes > 0

    again = cache.get_or_prepare(nodes, z, ENV, tb, max_neighbors=4, device=torch.device("cpu"))
    assert again is prepared
    assert cache.cache_hits >= 1


def test_block_static_topology_and_profiler():
    nodes = _nodes(8)
    block = MathRoutedTransformerBlock(
        d_model=32, n_heads=2, d_ff=64, dropout=0.0,
        attention_mode="neighbor_sparse", max_neighbors=4,
        topology_mode="middle_preserving_topk", fixed_k=4,
        middle_bridge_width=1, triton_block_d=16, triton_block_k=16,
    )
    block.prepare_static_topology(nodes, ENV, device=torch.device("cpu"))
    assert block.static_neighbors.shape == (8, 4)
    assert block.static_valid_i8.shape == (8, 4)
    assert block.attn.triton_block_d == 16
    assert block.attn.triton_block_k == 16

    x = torch.randn(1, 8, 32)
    block.eval()
    with torch.no_grad():
        out, timings = block.profile_cached_sparse_block(x, nodes, env=ENV)
    assert out.shape == x.shape
    for key in ["topology_prepare_ms", "qkv_ms", "attention_kernel_ms", "out_proj_ms", "ffn_ms", "total_block_ms"]:
        assert key in timings
        assert timings[key] >= 0.0


def test_block_static_fast_path_matches_cached_fast_path():
    nodes = _nodes(8)
    block = MathRoutedTransformerBlock(
        d_model=32, n_heads=2, d_ff=64, dropout=0.0,
        attention_mode="neighbor_sparse", max_neighbors=4,
        topology_mode="middle_preserving_topk", fixed_k=4,
        middle_bridge_width=1,
    )
    block.prepare_static_topology(nodes, ENV, device=torch.device("cpu"))
    x = torch.randn(1, 8, 32)
    block.eval()
    with torch.no_grad():
        cached = block.forward_cached_fast_path(x, nodes, env=ENV)
        static = block.forward_static_fast_path(x)
    assert torch.allclose(static, cached, atol=1e-6, rtol=1e-6)


def test_transformer_static_fast_path_after_prepare():
    nodes = _nodes(8)
    model = MathRoutedTransformer(
        d_model=32, n_heads=2, n_layers=2, d_ff=64, dropout=0.0,
        attention_mode="neighbor_sparse", max_neighbors=4,
        topology_mode="middle_preserving_topk", fixed_k=4,
        middle_bridge_width=1,
    )
    model.eval()
    model.prepare_static_topology(nodes, ENV, device=torch.device("cpu"))
    x = model.embed_nodes(nodes)
    with torch.no_grad():
        out = model.forward_static_fast_path(x)
    assert out.shape == x.shape
