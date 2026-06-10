from __future__ import annotations

import pytest
import torch

from src.attention import math_attention
from src.block_sparse_attention import (
    block_neighbors_to_dense_mask,
    block_sparse_attention_fast,
    block_sparse_attention_reference,
)


def _case(n: int, block_size: int, kb: int = 2):
    torch.manual_seed(n + block_size + kb)
    b = max(1, (n + block_size - 1) // block_size)
    neighbors = torch.zeros(b, kb, dtype=torch.long)
    valid = torch.zeros(b, kb, dtype=torch.int8)
    for i in range(b):
        vals = [i]
        if kb > 1:
            vals.append(min(b - 1, i + 1))
        for j, val in enumerate(vals[:kb]):
            neighbors[i, j] = val
            valid[i, j] = 1
    return neighbors, valid


@pytest.mark.parametrize("n", [1, 63, 64, 65, 127, 128, 129, 1024])
def test_block_sparse_reference_matches_dense_mask_cpu(n: int):
    block_size = 64
    q = torch.randn(2, 3, n, 8)
    k = torch.randn(2, 3, n, 8)
    v = torch.randn(2, 3, n, 8)
    nb, valid = _case(n, block_size, kb=2)

    sparse = block_sparse_attention_reference(q, k, v, nb, valid, block_size)
    mask = block_neighbors_to_dense_mask(nb, valid, block_size, n, device=q.device)
    dense = math_attention(q, k, v, mask.unsqueeze(0).unsqueeze(0))

    assert sparse.shape == q.shape
    assert not torch.isnan(sparse).any()
    assert torch.allclose(sparse, dense, atol=1e-5, rtol=1e-5)


def test_invalid_block_entries_are_masked():
    n = 129
    q = torch.randn(1, 1, n, 4)
    k = torch.randn(1, 1, n, 4)
    v = torch.randn(1, 1, n, 4)
    nb = torch.tensor([[0, 99, 1], [1, -1, 2], [2, 0, 7]])
    valid = torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 1]], dtype=torch.int8)
    sparse = block_sparse_attention_reference(q, k, v, nb, valid, 64)
    mask = block_neighbors_to_dense_mask(nb, valid, 64, n, device=q.device)
    dense = math_attention(q, k, v, mask.unsqueeze(0).unsqueeze(0))
    assert torch.allclose(sparse, dense, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("n,kb", [(63, 1), (128, 1), (129, 2), (1024, 3)])
def test_fast_entry_matches_reference_cpu(n: int, kb: int):
    q = torch.randn(1, 2, n, 8)
    k = torch.randn(1, 2, n, 8)
    v = torch.randn(1, 2, n, 8)
    nb, valid = _case(n, 64, kb=kb)
    ref = block_sparse_attention_reference(q, k, v, nb, valid, 64)
    fast = block_sparse_attention_fast(q, k, v, nb, valid, 64)
    assert torch.allclose(fast, ref, atol=1e-5, rtol=1e-5)


def test_block_sparse_reference_n4096_no_nan_cpu():
    n = 4096
    q = torch.randn(1, 1, n, 4)
    k = torch.randn(1, 1, n, 4)
    v = torch.randn(1, 1, n, 4)
    nb, valid = _case(n, 64, kb=1)
    out = block_sparse_attention_reference(q, k, v, nb, valid, 64)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()
