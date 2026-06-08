import torch
import pytest
from src.attention import math_attention
from src.sparse_attention import (
    neighbors_from_mask,
    neighbor_attention,
    max_k_from_mask,
    neighbors_from_candidate_qk_scores,
    neighbors_from_qk_scores,
    symbolic_priority_scores,
)


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


def test_neighbors_from_qk_scores_keeps_self_and_shape():
    torch.manual_seed(10)
    B, H, T, D = 1, 2, 5, 4
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)

    nb, valid = neighbors_from_qk_scores(q, k, max_k=3)

    assert nb.shape == (T, 3)
    assert valid.shape == (T, 3)
    assert valid.all()
    for i in range(T):
        assert i in nb[i].tolist()


def test_symbolic_kmip_bonus_changes_selection():
    B, H, T, D = 1, 1, 4, 2
    q = torch.zeros(B, H, T, D)
    k = torch.zeros(B, H, T, D)
    q[:, :, :, 0] = 1.0
    k[:, :, :, 0] = 1.0

    priority = torch.zeros(T, T, dtype=torch.int8)
    priority.fill_diagonal_(1)
    priority[0, 2] = 2
    symbolic = symbolic_priority_scores(priority)

    nb, valid = neighbors_from_qk_scores(
        q, k, max_k=2,
        symbolic_scores=symbolic,
        alpha=0.0,
        beta=1.0,
    )

    assert valid[0].all()
    assert set(nb[0].tolist()) == {0, 2}


def test_candidate_kmip_selects_only_from_candidates():
    B, H, T, D = 1, 1, 5, 2
    q = torch.zeros(B, H, T, D)
    k = torch.zeros(B, H, T, D)
    q[:, :, :, 0] = 1.0
    k[:, :, :, 0] = 1.0

    candidates = torch.tensor([
        [0, 2, 4],
        [1, 0, 3],
        [2, 1, 4],
        [3, 0, 2],
        [4, 1, 3],
    ])
    valid = torch.ones_like(candidates, dtype=torch.bool)
    symbolic = torch.zeros(T, T)
    symbolic[0, 4] = 10.0
    symbolic[0, 3] = 20.0  # Higher score, but not a row-0 candidate.

    nb, vld = neighbors_from_candidate_qk_scores(
        q, k,
        candidate_neighbors=candidates,
        candidate_valid=valid,
        max_k=2,
        symbolic_scores=symbolic,
        alpha=0.0,
        beta=1.0,
    )

    assert nb.shape == (T, 2)
    assert vld.shape == (T, 2)
    assert set(nb[0].tolist()) == {0, 4}
    assert 3 not in nb[0].tolist()


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


@pytest.mark.parametrize("selector", ["kmip_only", "symbolic_kmip", "symbolic_candidate_kmip"])
def test_neighbor_sparse_selector_modes_in_model_block(selector):
    from src.model import MathRoutedTransformerBlock
    from src.parser import parse
    from src.normalize import normalize

    torch.manual_seed(11)
    T, D = 8, 32
    nodes = [normalize(parse("add(matmul(A, x), b)")) for _ in range(T)]
    x = torch.randn(1, T, D)

    block = MathRoutedTransformerBlock(
        d_model=D, n_heads=4, d_ff=64, topk=2, local_window=1,
        attention_mode="neighbor_sparse", max_neighbors=6,
        sparse_selector=selector,
        selector_k=4 if selector == "symbolic_candidate_kmip" else None,
    )

    with torch.no_grad():
        out, mask, routes, diag = block(x, nodes)

    assert out.shape == (1, T, D)
    assert mask is not None
    assert routes is not None
    assert diag is not None


def test_prioritized_zero_priority_sorts_last_under_truncation():
    import numpy as np
    from src.sparse_attention import neighbors_from_mask_prioritized

    mask = torch.ones(1, 4, dtype=torch.bool)
    priority = np.array([[0, 7, 1, 2]], dtype=np.int8)
    nb, valid = neighbors_from_mask_prioritized(mask, priority, max_k=3)

    assert valid[0].all()
    assert nb[0].tolist() == [2, 3, 1]
