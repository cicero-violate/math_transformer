from __future__ import annotations
import math
import torch
import torch.nn.functional as F
import numpy as np


def neighbors_from_mask(
    mask: torch.Tensor,  # (T, T) bool
    max_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a boolean attention mask to padded neighbor lists.

    Returns
    -------
    neighbors : (T, max_k) LongTensor — neighbor indices (padded with self)
    valid     : (T, max_k) BoolTensor — True for real neighbors, False for padding
    """
    T = mask.shape[0]
    neighbors = torch.zeros(T, max_k, dtype=torch.long)
    valid = torch.zeros(T, max_k, dtype=torch.bool)
    for i in range(T):
        row_idx = mask[i].nonzero(as_tuple=False).view(-1)
        k = min(int(row_idx.numel()), max_k)
        if k > 0:
            neighbors[i, :k] = row_idx[:k]
            valid[i, :k] = True
        # Pad remaining with self-index (will be masked out via `valid`)
        neighbors[i, k:] = i
    return neighbors, valid


def neighbors_from_mask_prioritized(
    mask: torch.Tensor,           # (T, T) bool
    priority: np.ndarray | None,  # (T, T) int8 — lower value = higher priority; 0 = no edge
    max_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Like neighbors_from_mask but sorts each row's allowed neighbors by priority
    (ascending priority value = most important first) before truncating to max_k.

    Parameters
    ----------
    mask     : (T, T) bool — allowed positions
    priority : (T, T) int8 — relation priority per cell (0 = disconnected)
    max_k    : maximum number of neighbors to keep per row

    Returns
    -------
    neighbors : (T, max_k) LongTensor
    valid     : (T, max_k) BoolTensor
    """
    T = mask.shape[0]
    neighbors = torch.zeros(T, max_k, dtype=torch.long)
    valid = torch.zeros(T, max_k, dtype=torch.bool)

    for i in range(T):
        row_idx = mask[i].nonzero(as_tuple=False).view(-1)
        n_nbrs = row_idx.numel()
        if n_nbrs == 0:
            neighbors[i, :] = i
            continue

        if priority is not None:
            prow = priority[i]  # (T,) int8
            # Sort by priority ascending; ties broken by index
            nbrs_np = row_idx.numpy()
            prios = prow[nbrs_np]
            order = np.argsort(prios, kind="stable")
            row_idx_sorted = row_idx[torch.from_numpy(order)]
        else:
            row_idx_sorted = row_idx

        k = min(n_nbrs, max_k)
        neighbors[i, :k] = row_idx_sorted[:k]
        valid[i, :k] = True
        neighbors[i, k:] = i
    return neighbors, valid


def neighbor_attention(
    q: torch.Tensor,          # (B, H, T, D)
    k: torch.Tensor,          # (B, H, T, D)
    v: torch.Tensor,          # (B, H, T, D)
    neighbors: torch.Tensor,  # (T, K) — neighbor indices
    valid: torch.Tensor | None = None,  # (T, K) bool — True = real neighbor
) -> torch.Tensor:
    """
    Index-gather sparse attention.
    Computes scores only for the K neighbors of each token.
    Complexity: O(T * K * D)  vs  O(T^2 * D) for dense.
    """
    B, H, T, D = q.shape
    K = neighbors.shape[1]
    nb = neighbors.to(q.device)       # (T, K)

    # Gather K, V for each token's neighbors: (B, H, T, K, D)
    nb_flat = nb.reshape(-1)           # (T*K,)
    k_nb = k.index_select(2, nb_flat).view(B, H, T, K, D)
    v_nb = v.index_select(2, nb_flat).view(B, H, T, K, D)

    # Dot-product scores: (B, H, T, 1, D) * (B, H, T, K, D) → (B, H, T, K)
    scores = (q.unsqueeze(3) * k_nb).sum(-1) / math.sqrt(D)

    # Mask padding positions
    if valid is not None:
        val = valid.to(q.device)  # (T, K)
        scores = scores.masked_fill(
            ~val.unsqueeze(0).unsqueeze(0),  # (1, 1, T, K)
            float("-inf"),
        )

    probs = F.softmax(scores, dim=-1)          # (B, H, T, K)
    probs = torch.nan_to_num(probs, nan=0.0)  # handle all-masked rows

    # Weighted sum: (B, H, T, K, 1) * (B, H, T, K, D) → (B, H, T, D)
    out = (probs.unsqueeze(-1) * v_nb).sum(-2)
    return out


def neighbors_from_priority_torch(
    priority: torch.Tensor,  # (T, T) int8 on any device; 0=disconnected, 1-7=priority
    max_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GPU-native equivalent of neighbors_from_mask_prioritized.
    Sort each row by priority ascending (0=disconnected pushed past end), keep top max_k.
    Always returns (T, max_k) tensors, padding with self-index when max_k > T.

    Returns
    -------
    neighbors : (T, max_k) LongTensor
    valid     : (T, max_k) BoolTensor
    """
    T = priority.shape[0]
    device = priority.device

    sort_key = priority.to(torch.float32)
    sort_key = sort_key.masked_fill(priority == 0, 256.0)

    _, sorted_idx = torch.sort(sort_key, dim=1, stable=True)  # (T, T)

    # Expand to (T, max_k): take min(max_k, T) from sorted, pad rest with self-index
    take = min(max_k, T)
    top_idx = sorted_idx[:, :take]  # (T, take)

    if take < max_k:
        self_pad = torch.arange(T, device=device).unsqueeze(1).expand(T, max_k - take)
        top_idx = torch.cat([top_idx, self_pad], dim=1)  # (T, max_k)

    # Check validity only for the real (non-padded) positions
    top_prio = priority.to(torch.int32).gather(1, top_idx)  # (T, max_k)
    valid = top_prio > 0
    if take < max_k:
        valid[:, take:] = False  # padding slots are never valid

    self_idx = torch.arange(T, device=device).unsqueeze(1).expand_as(top_idx)
    neighbors = torch.where(valid, top_idx, self_idx)

    return neighbors, valid


def symbolic_priority_scores(
    priority: torch.Tensor,
) -> torch.Tensor:
    """
    Convert relation priority to an additive symbolic score.

    Priority uses lower integers for stronger symbolic relations and 0 for no
    relation. The returned score uses larger values for stronger relations and
    leaves disconnected pairs at 0 so learned k-MIP scores can still recover
    neighbors outside the static topology.
    """
    priority_i16 = priority.to(torch.int16)
    return torch.where(
        priority_i16 > 0,
        8 - priority_i16,
        torch.zeros((), dtype=torch.int16, device=priority.device),
    ).to(torch.float32)


def neighbors_from_qk_scores(
    q: torch.Tensor,  # (B, H, T, D)
    k: torch.Tensor,  # (B, H, T, D)
    max_k: int,
    symbolic_scores: torch.Tensor | None = None,  # (T, T), larger = better
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a shared top-K neighbor table from learned Q/K geometry.

    The Triton sparse kernel consumes one neighbor table shared across batch and
    heads, so this averages scaled QK scores across (B, H) before selecting
    neighbors. If symbolic_scores is provided, selection uses:

        alpha * mean(q_i dot k_j / sqrt(D)) + beta * symbolic_scores[i, j]
    """
    if max_k <= 0:
        raise ValueError("max_k must be positive")
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {q.shape} and {k.shape}")

    B, H, T, D = q.shape
    kk = min(max_k, T)
    learned = torch.einsum("bhtd,bhsd->bhts", q, k).mean(dim=(0, 1))
    learned = learned * (1.0 / math.sqrt(D))
    scores = learned * alpha

    if symbolic_scores is not None:
        if symbolic_scores.shape != (T, T):
            raise ValueError(
                f"symbolic_scores must have shape {(T, T)}, got {tuple(symbolic_scores.shape)}"
            )
        scores = scores + beta * symbolic_scores.to(device=q.device, dtype=scores.dtype)

    # Keep self-attention even when learned scores are low.
    diag_boost = torch.finfo(scores.dtype).max / 4
    idx = torch.arange(T, device=q.device)
    scores = scores.clone()
    scores[idx, idx] = diag_boost

    top_idx = scores.topk(kk, dim=1).indices
    valid = torch.ones(T, kk, dtype=torch.bool, device=q.device)

    if kk < max_k:
        pad = idx.unsqueeze(1).expand(T, max_k - kk)
        top_idx = torch.cat([top_idx, pad], dim=1)
        valid = torch.cat(
            [valid, torch.zeros(T, max_k - kk, dtype=torch.bool, device=q.device)],
            dim=1,
        )

    return top_idx.long(), valid


def neighbors_from_candidate_qk_scores(
    q: torch.Tensor,  # (B, H, T, D)
    k: torch.Tensor,  # (B, H, T, D)
    candidate_neighbors: torch.Tensor,  # (T, C)
    candidate_valid: torch.Tensor | None,
    max_k: int,
    symbolic_scores: torch.Tensor | None = None,  # (T, T), larger = better
    alpha: float = 1.0,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select final top-K neighbors from a cached symbolic candidate table.

    Unlike neighbors_from_qk_scores, this never scores all T*T pairs. It scores
    only T*C candidate slots, where C is the cached symbolic candidate budget.
    """
    if max_k <= 0:
        raise ValueError("max_k must be positive")
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {q.shape} and {k.shape}")

    B, H, T, D = q.shape
    if candidate_neighbors.shape[0] != T:
        raise ValueError(
            f"candidate_neighbors must have {T} rows, got {candidate_neighbors.shape[0]}"
        )

    candidates = candidate_neighbors.to(device=q.device, dtype=torch.long)
    C = candidates.shape[1]
    kk = min(max_k, C)

    flat = candidates.reshape(-1)
    k_nb = k.index_select(2, flat).view(B, H, T, C, D)
    learned = (q.unsqueeze(3) * k_nb).sum(dim=-1).mean(dim=(0, 1))
    scores = learned * (alpha / math.sqrt(D))

    if symbolic_scores is not None:
        if symbolic_scores.shape != (T, T):
            raise ValueError(
                f"symbolic_scores must have shape {(T, T)}, got {tuple(symbolic_scores.shape)}"
            )
        sym = symbolic_scores.to(device=q.device, dtype=scores.dtype).gather(1, candidates)
        scores = scores + beta * sym

    if candidate_valid is not None:
        valid_candidates = candidate_valid.to(device=q.device, dtype=torch.bool)
        scores = scores.masked_fill(~valid_candidates, -torch.inf)
    else:
        valid_candidates = torch.ones(T, C, dtype=torch.bool, device=q.device)

    rows = torch.arange(T, device=q.device).unsqueeze(1)
    self_slots = candidates == rows
    scores = scores.masked_fill(self_slots, torch.finfo(scores.dtype).max / 4)

    top_pos = scores.topk(kk, dim=1).indices
    neighbors = candidates.gather(1, top_pos)
    valid = valid_candidates.gather(1, top_pos)

    if kk < max_k:
        self_pad = torch.arange(T, device=q.device).unsqueeze(1).expand(T, max_k - kk)
        neighbors = torch.cat([neighbors, self_pad], dim=1)
        valid = torch.cat(
            [valid, torch.zeros(T, max_k - kk, dtype=torch.bool, device=q.device)],
            dim=1,
        )

    return neighbors.long(), valid


def max_k_from_mask(mask: torch.Tensor) -> int:
    """Return the maximum number of allowed neighbors across all rows."""
    return int(mask.sum(dim=-1).max().item())


# Sprint 3: torch.compile wrapper compiled once at import time.
try:
    neighbor_attention_compiled = torch.compile(neighbor_attention)
    _COMPILED_AVAILABLE = True
except Exception:
    neighbor_attention_compiled = neighbor_attention
    _COMPILED_AVAILABLE = False
