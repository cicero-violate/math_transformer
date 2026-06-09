import torch
import pytest
from src.attention import math_attention, MathRoutedAttention, NeighborSparseMathAttention
from src.sparse_attention import neighbors_from_mask, max_k_from_mask


def test_math_attention_shape_no_mask():
    B, T, D = 2, 8, 16
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)
    out = math_attention(q, k, v)
    assert out.shape == (B, T, D)


def test_math_attention_shape_with_mask():
    B, T, D = 1, 6, 8
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)
    mask = torch.ones(T, T, dtype=torch.bool)
    out = math_attention(q, k, v, mask)
    assert out.shape == (B, T, D)


def test_masked_attention_matches_full_when_all_allowed():
    torch.manual_seed(0)
    B, T, D = 1, 4, 8
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)
    full = math_attention(q, k, v, None)
    all_mask = torch.ones(T, T, dtype=torch.bool)
    masked = math_attention(q, k, v, all_mask)
    assert torch.allclose(full, masked, atol=1e-5)


def test_diagonal_mask_reduces_context():
    torch.manual_seed(1)
    B, T, D = 1, 8, 16
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)
    full = math_attention(q, k, v, None)
    diag_mask = torch.eye(T, dtype=torch.bool)
    sparse = math_attention(q, k, v, diag_mask)
    assert sparse.shape == full.shape
    # Diagonal-only should differ from full attention
    assert not torch.allclose(full, sparse, atol=1e-4)


def test_routed_attention_module_shape():
    B, T, D = 2, 6, 32
    attn = MathRoutedAttention(D, n_heads=4)
    x = torch.randn(B, T, D)
    out = attn(x)
    assert out.shape == (B, T, D)


def test_routed_attention_with_mask():
    B, T, D = 1, 5, 32
    attn = MathRoutedAttention(D, n_heads=4)
    x = torch.randn(B, T, D)
    mask = torch.ones(T, T, dtype=torch.bool)
    out = attn(x, mask)
    assert out.shape == (B, T, D)


# ── NeighborSparseMathAttention module tests ──────────────────────────────────

def test_neighbor_sparse_module_shape():
    torch.manual_seed(10)
    B, T, D = 1, 8, 32
    mask = torch.rand(T, T) < 0.5
    mask.fill_diagonal_(True)
    K = max_k_from_mask(mask)
    nb, valid = neighbors_from_mask(mask, K)

    attn = NeighborSparseMathAttention(D, n_heads=4)
    x = torch.randn(B, T, D)
    out = attn(x, nb, valid)
    assert out.shape == (B, T, D)


def test_neighbor_sparse_module_matches_dense_masked():
    """NeighborSparseMathAttention must match DenseMaskedMathAttention for same mask."""
    import torch.nn as nn
    from src.attention import DenseMaskedMathAttention

    torch.manual_seed(0)
    B, T, D = 1, 6, 32
    mask = torch.rand(T, T) < 0.5
    mask.fill_diagonal_(True)
    K = max_k_from_mask(mask)
    nb, valid = neighbors_from_mask(mask, K)

    # Share weights
    dense_attn = DenseMaskedMathAttention(D, n_heads=4)
    sparse_attn = NeighborSparseMathAttention(D, n_heads=4)

    with torch.no_grad():
        for pname in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            p_dense = dict(dense_attn.named_parameters())[pname + ".weight"]
            p_sparse = dict(sparse_attn.named_parameters())[pname + ".weight"]
            p_sparse.copy_(p_dense)
            if pname == "out_proj":
                pb_d = dense_attn.out_proj.bias
                pb_s = sparse_attn.out_proj.bias
                pb_s.copy_(pb_d)

    x = torch.randn(B, T, D)
    with torch.no_grad():
        out_dense = dense_attn(x, mask)
        out_sparse = sparse_attn(x, nb, valid)

    assert torch.allclose(out_dense, out_sparse, atol=1e-4), (
        f"Max diff: {(out_dense - out_sparse).abs().max().item():.2e}"
    )
