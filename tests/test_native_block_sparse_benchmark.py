from __future__ import annotations

import torch

from src.block_learned_topology import HeuristicBlockTopologyBuilder
from src.ir import add, matmul, var
from src.model import MathRoutedTransformerBlock


def _nodes(n: int):
    expr = add(matmul(var("A"), var("x")), var("b"))
    base = expr.collect_nodes()
    return [base[i % len(base)] for i in range(n)]


def test_block_sparse_model_static_path_runs_cpu():
    nodes = _nodes(129)
    block = MathRoutedTransformerBlock(
        d_model=32,
        n_heads=2,
        d_ff=64,
        dropout=0.0,
        attention_mode="block_sparse",
        max_neighbors=8,
    )
    block.topology = HeuristicBlockTopologyBuilder(
        block_size=64,
        topk_blocks=2,
        block_token_cap=4,
        fixed_k=8,
        device="cpu",
    )
    block.eval()
    prepared = block.prepare_static_topology(nodes, device=torch.device("cpu"))
    assert prepared.is_block_topology is True
    x = torch.randn(1, 129, 32)
    with torch.no_grad():
        out = block.forward_static_fast_path(x)
        _, timings = block.profile_cached_sparse_block(x, nodes)
    assert out.shape == x.shape
    assert timings["attention_kernel_ms"] >= 0.0
    assert timings["total_block_ms"] >= timings["attention_kernel_ms"]
