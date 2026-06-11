from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .embedder import MathEmbedder, pairwise_cosine
from .ir import MathNode
from .middle_preserving_topology import middle_bridge_matrix
from .topology import (
    composition_matrix,
    identity_matrix,
    local_window_matrix,
    same_operator_matrix,
    shape_compatibility_matrix,
    symbolic_dependency_matrix,
)


DEFAULT_NONCOMMUTATIVE_OPS: tuple[str, ...] = ("sub", "div", "matmul")


FEATURE_NAMES: tuple[str, ...] = (
    "identity",
    "symbolic_dependency",
    "composition",
    "shape_compat",
    "middle_bridge",
    "local_window",
    "same_operator",
    "embedding_cos",
    "relative_abs_position",
    "relative_signed_position",
)


@dataclass(frozen=True)
class EdgeFeatureSpec:
    names: tuple[str, ...] = FEATURE_NAMES

    @property
    def dim(self) -> int:
        return len(self.names)


def build_edge_feature_tensor(
    nodes: list[MathNode],
    z: np.ndarray | None = None,
    env: dict[str, tuple[int, ...]] | None = None,
    *,
    local_window: int = 1,
    middle_bridge_width: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build dense pairwise edge features for a learned topology scorer.

    This is intentionally offline/training-friendly. The production hot path should
    consume compiled/cached TopK neighbor tables, not rebuild this tensor per block.
    """
    n = len(nodes)
    if z is None:
        z = MathEmbedder().encode_batch(nodes)

    ident = identity_matrix(n).astype(np.float32)
    sym = symbolic_dependency_matrix(nodes).astype(np.float32)
    comp = composition_matrix(nodes, env).astype(np.float32)
    shape = shape_compatibility_matrix(nodes, env).astype(np.float32)
    middle = middle_bridge_matrix(n, middle_bridge_width).astype(np.float32)
    local = local_window_matrix(n, local_window).astype(np.float32)
    sameop = same_operator_matrix(nodes).astype(np.float32)
    cos = pairwise_cosine(z).clip(0.0, 1.0).astype(np.float32)
    if n > 0:
        np.fill_diagonal(cos, 0.0)

    if n <= 1:
        rel_abs = np.zeros((n, n), dtype=np.float32)
        rel_signed = np.zeros((n, n), dtype=np.float32)
    else:
        idx = np.arange(n, dtype=np.float32)
        rel_signed = (idx[None, :] - idx[:, None]) / float(n - 1)
        rel_abs = np.abs(rel_signed).astype(np.float32)
        rel_signed = rel_signed.astype(np.float32)

    features = np.stack(
        [ident, sym, comp, shape, middle, local, sameop, cos, rel_abs, rel_signed],
        axis=-1,
    ).astype(np.float32)
    return torch.tensor(features, dtype=torch.float32, device=device)


class LearnedTopologyScorer(nn.Module):
    """Small MLP edge scorer for learned sparse topology experiments."""

    def __init__(self, feature_dim: int = len(FEATURE_NAMES), hidden_dim: int = 64) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        if edge_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.feature_dim}, got {edge_features.shape[-1]}"
            )
        return self.net(edge_features).squeeze(-1)


def topk_mask_from_scores(
    scores: torch.Tensor,
    fixed_k: int,
    *,
    force_self: bool = True,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert learned edge scores to a per-row fixed-K boolean mask."""
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError(f"scores must be square (T,T), got {tuple(scores.shape)}")
    if fixed_k <= 0:
        raise ValueError("fixed_k must be positive")
    n = scores.shape[0]
    k = min(fixed_k, n)
    work = scores.clone()
    if candidate_mask is not None:
        if candidate_mask.shape != scores.shape:
            raise ValueError(
                f"candidate_mask must have shape {tuple(scores.shape)}, got {tuple(candidate_mask.shape)}"
            )
        work = work.masked_fill(~candidate_mask.to(device=scores.device, dtype=torch.bool), -torch.inf)
    if force_self and n > 0:
        idx = torch.arange(n, device=scores.device)
        work[idx, idx] = torch.finfo(work.dtype).max / 4
    top_idx = work.topk(k, dim=1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    rows = torch.arange(n, device=scores.device).unsqueeze(1).expand_as(top_idx)
    mask[rows.reshape(-1), top_idx.reshape(-1)] = True
    if force_self and n > 0:
        mask.fill_diagonal_(True)
    return mask


def learned_topk_mask(
    scorer: LearnedTopologyScorer,
    nodes: list[MathNode],
    z: np.ndarray | None = None,
    env: dict[str, tuple[int, ...]] | None = None,
    *,
    fixed_k: int = 8,
    local_window: int = 1,
    middle_bridge_width: int = 0,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(mask, scores)` for the current learned scorer."""
    dev = torch.device(device) if device is not None else next(scorer.parameters()).device
    features = build_edge_feature_tensor(
        nodes,
        z,
        env,
        local_window=local_window,
        middle_bridge_width=middle_bridge_width,
        device=dev,
    )
    scores = scorer(features)
    return topk_mask_from_scores(scores, fixed_k), scores