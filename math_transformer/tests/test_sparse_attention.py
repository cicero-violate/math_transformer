import torch
import pytest
from src.attention import math_attention
from src.sparse_attention import neighbors_from_mask, neighbor_attention, max_k_from_mask


def _make_mask(T: int, density: float = 0.4) -> torch.Tensor:
    """Random boolean mask with given density plus self-attention."""
    torch.manual_seed(42)
    mask = torch.rand(T, T) < density
    mask.fill_diagonal_(True)
    return mask


def test_neighbors_from_mask_shape():
    T, K = 6, 3
    mask = torch.ones(T, T, dtype=torch.bool)
    nb, valid = neighbors_from_mask(mask, K)
    assert nb.shape == (T, K)
    assert valid.shape == (T, K)


def test_neighbors_from_mask_valid_count():
    T = 4
    mask = torch.eye(T, dtype=torch.bool)  # only self-attention
    nb, valid = neighbors_from_mask(mask, max_k=4)
    # Each row has exactly 1 valid neighbor (self)
    assert valid.sum(dim=-1).tolist() == [1, 1, 1, 1]


def test_neighbors_are_correct_indices():
    T = 4
    mask = torch.zeros(T, T, dtype=torch.bool)
    mask[0, 0] = mask[0, 2] = True   # row 0: neighbors 0, 2
    mask[1, 1] = mask[1, 3] = True   # row 1: neighbors 1, 3
    nb, valid = neighbors_from_mask(mask, max_k=3)
    row0 = sorted(nb[0, valid[0]].tolist())
    assert row0 == [0, 2]
    row1 = sorted(nb[1, valid[1]].tolist())
    assert row1 == [1, 3]


def test_max_k_from_mask():
    T = 5
    mask = torch.zeros(T, T, dtype=torch.bool)
    mask[0, :3] = True   # row 0 has 3 neighbors
    mask[1, :2] = True
    assert max_k_from_mask(mask) == 3


def test_neighbor_attention_shape():
    B, H, T, D = 1, 2, 6, 16
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    mask = _make_mask(T)
    K = max_k_from_mask(mask)
    nb, valid = neighbors_from_mask(mask, K)
    out = neighbor_attention(q, k, v, nb, valid)
    assert out.shape == (B, H, T, D)


def test_sparse_matches_dense_masked():
    """
    Core equivalence gate (Gate 3):
    neighbor_attention must match math_attention for the same mask.
    """
    torch.manual_seed(0)
    B, H, T, D = 1, 2, 8, 16
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    mask = _make_mask(T, density=0.5)
    K = max_k_from_mask(mask)
    nb, valid = neighbors_from_mask(mask, K)

    # Dense reference: expand mask to (B, H, T, T)
    mask_4d = mask.unsqueeze(0).unsqueeze(0)
    out_dense = math_attention(q, k, v, mask_4d)
    out_sparse = neighbor_attention(q, k, v, nb, valid)

    assert torch.allclose(out_dense, out_sparse, atol=1e-5), (
        f"Max diff: {(out_dense - out_sparse).abs().max().item():.2e}"
    )


def test_sparse_full_attention_matches_dense_full():
    """All-ones mask: sparse should match unconstrained dense."""
    torch.manual_seed(1)
    B, H, T, D = 1, 1, 5, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    mask = torch.ones(T, T, dtype=torch.bool)
    nb, valid = neighbors_from_mask(mask, T)

    out_dense = math_attention(q, k, v, None)
    out_sparse = neighbor_attention(q, k, v, nb, valid)

    assert torch.allclose(out_dense, out_sparse, atol=1e-5)


def test_sparse_single_neighbor_is_copy_of_v():
    """Self-only mask: output[i] should equal v[i]."""
    torch.manual_seed(2)
    B, H, T, D = 1, 1, 4, 8
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)

    mask = torch.eye(T, dtype=torch.bool)
    nb, valid = neighbors_from_mask(mask, 1)
    out = neighbor_attention(q, k, v, nb, valid)

    # With only self as neighbor, softmax gives 1.0 → output = v[self]
    assert torch.allclose(out, v, atol=1e-6)


# ── Prioritized neighbor tests ────────────────────────────────────────────────

def test_neighbors_from_mask_prioritized_shape():
    import numpy as np
    from src.sparse_attention import neighbors_from_mask_prioritized

    T, K = 6, 3
    mask = torch.rand(T, T) < 0.5
    mask.fill_diagonal_(True)
    priority = np.random.randint(1, 8, (T, T), dtype=np.int8)

    nb, valid = neighbors_from_mask_prioritized(mask, priority, K)
    assert nb.shape == (T, K)
    assert valid.shape == (T, K)


def test_prioritized_keeps_high_priority_first():
    """Identity self-edges (priority=1) must appear before same_op edges (priority=7)."""
    import numpy as np
    from src.sparse_attention import neighbors_from_mask_prioritized

    T = 4
    mask = torch.ones(T, T, dtype=torch.bool)
    priority = np.full((T, T), 7, dtype=np.int8)
    # Mark diagonal as priority 1
    np.fill_diagonal(priority, 1)

    nb, valid = neighbors_from_mask_prioritized(mask, priority, max_k=T)
    # The first column of every row should be the self-index
    for i in range(T):
        assert nb[i, 0].item() == i, f"Row {i}: expected self-index first, got {nb[i, 0].item()}"


# ── Model block integration tests ─────────────────────────────────────────────

def test_neighbor_sparse_in_model_block():
    """neighbor_sparse block forward must produce same shape as dense_masked."""
    from src.model import MathRoutedTransformerBlock
    from src.parser import parse
    from src.normalize import normalize

    torch.manual_seed(5)
    T, D = 8, 64
    nodes = [normalize(parse("add(matmul(A, x), b)")) for _ in range(T)]
    x = torch.randn(1, T, D)

    block_masked = MathRoutedTransformerBlock(
        d_model=D, n_heads=4, d_ff=128, topk=2, local_window=1,
        attention_mode="dense_masked"
    )
    block_sparse = MathRoutedTransformerBlock(
        d_model=D, n_heads=4, d_ff=128, topk=2, local_window=1,
        attention_mode="neighbor_sparse"
    )

    with torch.no_grad():
        out_m, mask_m, _, _ = block_masked(x, nodes)
        out_s, mask_s, _, _ = block_sparse(x, nodes)

    assert out_m.shape == out_s.shape == (1, T, D)


def test_neighbor_sparse_in_full_transformer():
    """Full MathRoutedTransformer with neighbor_sparse must produce correct output shape."""
    from src.model import MathRoutedTransformer
    from src.parser import parse
    from src.normalize import normalize

    torch.manual_seed(7)
    T, D = 8, 64
    nodes = [normalize(parse("matmul(Q, K)")) for _ in range(T)]

    model = MathRoutedTransformer(
        d_model=D, n_heads=4, n_layers=2, d_ff=128,
        topk=2, local_window=1, attention_mode="neighbor_sparse"
    )
    x = model.embed_nodes(nodes)

    with torch.no_grad():
        out, masks, routes = model(x, nodes)

    assert out.shape == (1, T, D)
    assert len(masks) == 2
