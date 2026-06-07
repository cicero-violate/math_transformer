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
