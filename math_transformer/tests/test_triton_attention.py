"""Tests for the Triton fused neighbor-sparse attention kernel (Sprint 5)."""
import math
import pytest
import torch

triton = pytest.importorskip("triton")

from src.triton_attention import triton_neighbor_attention, triton_neighbor_attention_flat, TRITON_AVAILABLE
from src.sparse_attention import neighbor_attention


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Triton kernel tests",
)

DEVICE = "cuda"


def _make_uniform_neighbors(T: int, K: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Each token i attends to the first K others (wrapping around, skipping self)."""
    neighbors = torch.zeros(T, K, dtype=torch.long, device=device)
    valid = torch.zeros(T, K, dtype=torch.bool, device=device)
    for t in range(T):
        others = [j for j in range(T) if j != t][:K]
        for ki, j in enumerate(others):
            neighbors[t, ki] = j
            valid[t, ki] = True
    return neighbors, valid


def _make_sparse_neighbors(T: int, K: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Random valid neighbors — some rows have fewer than K valid entries."""
    torch.manual_seed(0)
    neighbors = torch.randint(0, T, (T, K), device=device)
    valid = torch.rand(T, K, device=device) > 0.3
    # Self-loop for rows with no valid neighbors
    has_none = ~valid.any(dim=1)
    neighbors[has_none, 0] = torch.arange(T, device=device)[has_none]
    valid[has_none, 0] = True
    return neighbors, valid


def _ref(q, k, v, neighbors, valid):
    """PyTorch reference on the same device."""
    return neighbor_attention(q, k, v, neighbors, valid)


@pytest.mark.parametrize("B,H,T,D,K", [
    (1, 1, 8, 16, 4),
    (2, 4, 16, 32, 8),
    (1, 4, 32, 64, 16),
])
def test_triton_matches_pytorch_uniform(B, H, T, D, K):
    torch.manual_seed(42)
    q = torch.randn(B, H, T, D, device=DEVICE)
    k = torch.randn(B, H, T, D, device=DEVICE)
    v = torch.randn(B, H, T, D, device=DEVICE)
    neighbors, valid = _make_uniform_neighbors(T, min(K, T - 1), DEVICE)

    out_tri = triton_neighbor_attention(q, k, v, neighbors, valid)
    out_ref = _ref(q, k, v, neighbors, valid)

    assert out_tri.shape == out_ref.shape, "shape mismatch"
    assert torch.allclose(out_tri, out_ref, atol=1e-4, rtol=1e-4), (
        f"max diff: {(out_tri - out_ref).abs().max().item():.2e}"
    )


@pytest.mark.parametrize("B,H,T,D,K", [
    (1, 2, 12, 16, 6),
    (2, 4, 20, 32, 8),
])
def test_triton_matches_pytorch_sparse_valid(B, H, T, D, K):
    """Some valid flags are False — kernel must mask them out."""
    torch.manual_seed(7)
    q = torch.randn(B, H, T, D, device=DEVICE)
    k = torch.randn(B, H, T, D, device=DEVICE)
    v = torch.randn(B, H, T, D, device=DEVICE)
    neighbors, valid = _make_sparse_neighbors(T, K, DEVICE)

    out_tri = triton_neighbor_attention(q, k, v, neighbors, valid)
    out_ref = _ref(q, k, v, neighbors, valid)

    assert torch.allclose(out_tri, out_ref, atol=1e-4, rtol=1e-4), (
        f"max diff: {(out_tri - out_ref).abs().max().item():.2e}"
    )


def test_triton_output_shape():
    B, H, T, D, K = 2, 4, 16, 64, 8
    q = torch.randn(B, H, T, D, device=DEVICE)
    k = torch.randn(B, H, T, D, device=DEVICE)
    v = torch.randn(B, H, T, D, device=DEVICE)
    neighbors, valid = _make_uniform_neighbors(T, K, DEVICE)
    out = triton_neighbor_attention(q, k, v, neighbors, valid)
    assert out.shape == (B, H, T, D)


def test_triton_flat_matches_bhtd_wrapper():
    B, H, T, D, K = 2, 4, 16, 32, 8
    torch.manual_seed(11)
    q = torch.randn(B, H, T, D, device=DEVICE)
    k = torch.randn(B, H, T, D, device=DEVICE)
    v = torch.randn(B, H, T, D, device=DEVICE)
    neighbors, valid = _make_uniform_neighbors(T, K, DEVICE)
    out_bhtd = triton_neighbor_attention(q, k, v, neighbors, valid)
    out_flat = triton_neighbor_attention_flat(q, k, v, neighbors, valid.char())
    expected = out_bhtd.transpose(1, 2).reshape(B, T, H * D)
    assert out_flat.shape == expected.shape
    assert torch.allclose(out_flat, expected, atol=1e-4, rtol=1e-4)


def test_triton_all_invalid_row_gives_zero():
    """A token with no valid neighbors should produce an all-zero output."""
    B, H, T, D, K = 1, 1, 4, 16, 3
    q = torch.randn(B, H, T, D, device=DEVICE)
    k = torch.randn(B, H, T, D, device=DEVICE)
    v = torch.randn(B, H, T, D, device=DEVICE)
    neighbors = torch.zeros(T, K, dtype=torch.long, device=DEVICE)
    valid = torch.zeros(T, K, dtype=torch.bool, device=DEVICE)
    # token 0 has no valid neighbors; tokens 1-3 attend to token 0
    valid[1:, 0] = True

    out = triton_neighbor_attention(q, k, v, neighbors, valid)
    assert out.shape == (B, H, T, D)
    # The all-invalid row (token 0) should be zero (softmax of all-inf → 0 weights)
    assert out[0, 0, 0].abs().max().item() < 1e-5, "all-invalid row should be ~zero"


def test_triton_single_neighbor_equals_value():
    """When each token has exactly one valid neighbor, output equals that V row."""
    B, H, T, D = 1, 1, 4, 16
    K = 1
    torch.manual_seed(3)
    q = torch.randn(B, H, T, D, device=DEVICE)
    k = torch.randn(B, H, T, D, device=DEVICE)
    v = torch.randn(B, H, T, D, device=DEVICE)
    # Each token i attends to token (i+1) % T
    target = torch.tensor([(i + 1) % T for i in range(T)], device=DEVICE)
    neighbors = target.unsqueeze(1)       # (T, 1)
    valid = torch.ones(T, K, dtype=torch.bool, device=DEVICE)

    out = triton_neighbor_attention(q, k, v, neighbors, valid)
    # With one neighbor, softmax gives prob=1, so output = V[neighbor]
    expected = v[0, 0][target]  # (T, D)
    assert torch.allclose(out[0, 0], expected, atol=1e-5), (
        f"max diff: {(out[0, 0] - expected).abs().max().item():.2e}"
    )
