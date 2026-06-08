from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


def middle_anchor_indices(n: int, width: int = 0) -> list[int]:
    """
    Return deterministic middle-context anchor indices.

    Anchors cover the center and quartiles. `width` expands each anchor by a
    small local band while preserving sorted uniqueness.
    """
    if n <= 0:
        return []
    width = max(int(width), 0)
    centers = [n // 4, n // 2, (3 * n) // 4]
    out: set[int] = set()
    for center in centers:
        for offset in range(-width, width + 1):
            j = center + offset
            if 0 <= j < n:
                out.add(j)
    return sorted(out)


def middle_bridge_matrix(n: int, width: int = 0, *, causal: bool = False) -> np.ndarray:
    """
    Boolean bridge matrix G where every row can route to deterministic middle
    anchors. When `causal=True`, future anchors j > i are suppressed.
    """
    mat = np.zeros((n, n), dtype=bool)
    anchors = middle_anchor_indices(n, width)
    if n <= 0 or not anchors:
        return mat
    for i in range(n):
        for j in anchors:
            if not causal or j <= i:
                mat[i, j] = True
    np.fill_diagonal(mat, True)
    return mat


def middle_bridge_matrix_torch(
    n: int,
    width: int = 0,
    *,
    device: torch.device | str = "cpu",
    causal: bool = False,
) -> torch.Tensor:
    """Torch equivalent of middle_bridge_matrix."""
    device = torch.device(device)
    mat = torch.zeros(n, n, dtype=torch.bool, device=device)
    anchors = middle_anchor_indices(n, width)
    if n <= 0 or not anchors:
        return mat
    rows = torch.arange(n, device=device)
    for j in anchors:
        if causal:
            mat[rows >= j, j] = True
        else:
            mat[:, j] = True
    mat.fill_diagonal_(True)
    return mat


def middle_band_indices(n: int, low: float = 0.375, high: float = 0.625) -> list[int]:
    """Return indices in the middle band used by the coverage diagnostic."""
    if n <= 0:
        return []
    lo = max(0, min(n - 1, int(np.floor(low * n))))
    hi = max(lo + 1, min(n, int(np.ceil(high * n))))
    return list(range(lo, hi))


def _coverage_from_rows(rows: Iterable[Iterable[int]], n: int) -> float:
    band = set(middle_band_indices(n))
    if n <= 0 or not band:
        return 0.0
    total = 0
    hit = 0
    for row in rows:
        vals = list(row)
        if not vals:
            continue
        total += 1
        if any(int(j) in band for j in vals):
            hit += 1
    return hit / total if total else 0.0


def middle_coverage_score(mask_or_neighbors: np.ndarray | torch.Tensor) -> float:
    """
    Fraction of rows with at least one selected middle-band neighbor.

    Accepts either:
    - a square boolean mask `(T, T)`; or
    - an integer neighbor table `(T, K)`.
    """
    x = mask_or_neighbors.detach().cpu().numpy() if isinstance(mask_or_neighbors, torch.Tensor) else np.asarray(mask_or_neighbors)
    if x.ndim != 2:
        raise ValueError(f"expected 2-D mask or neighbor table, got shape {x.shape}")
    n = int(x.shape[0])
    if x.dtype == bool and x.shape[0] == x.shape[1]:
        rows = (np.flatnonzero(x[i]).tolist() for i in range(n))
        return _coverage_from_rows(rows, n)
    return _coverage_from_rows((x[i].tolist() for i in range(n)), n)
