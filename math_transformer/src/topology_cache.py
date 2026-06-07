from __future__ import annotations
import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from .ir import MathNode
from .topology import MaskDiagnostics


@dataclass
class CachedTopology:
    mask: torch.Tensor         # (T, T) bool
    priority: np.ndarray       # (T, T) int8
    neighbors: torch.Tensor    # (T, K) long
    valid: torch.Tensor        # (T, K) bool
    diagnostics: MaskDiagnostics


def stable_nodes_hash(nodes: list[MathNode]) -> str:
    h = hashlib.sha256()
    for nd in nodes:
        h.update(repr(nd).encode())
    return h.hexdigest()


def stable_env_hash(env: dict[str, tuple[int, ...]] | None) -> str:
    if not env:
        return "no_env"
    serialized = ",".join(f"{k}:{v}" for k, v in sorted(env.items()))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _cache_key(
    nodes: list[MathNode],
    env: dict | None,
    topk: int,
    local_window: int,
    max_neighbors: int | None,
) -> str:
    return "|".join([
        stable_nodes_hash(nodes),
        stable_env_hash(env),
        str(topk),
        str(local_window),
        str(max_neighbors),
    ])


class TopologyCache:
    """
    In-process LRU-style cache for topology builds.
    Keyed on (nodes_hash, env_hash, topk, local_window, max_neighbors).
    """

    def __init__(self, maxsize: int = 256) -> None:
        self._store: dict[str, CachedTopology] = {}
        self._order: list[str] = []   # insertion-order eviction
        self.maxsize = maxsize
        self.cache_hits = 0
        self.cache_misses = 0

    def get_or_build(
        self,
        nodes: list[MathNode],
        z: np.ndarray,
        env: dict | None,
        builder,                      # TopologyBuilder instance
        max_neighbors: int | None = None,
    ) -> CachedTopology:
        from .topology import build_priority_matrix
        from .sparse_attention import neighbors_from_mask_prioritized

        key = _cache_key(nodes, env, builder.topk, builder.local_window, max_neighbors)

        if key in self._store:
            self.cache_hits += 1
            return self._store[key]

        self.cache_misses += 1

        np_mask, diag = builder.build_detailed(nodes, z, env)
        mask_t = torch.tensor(np_mask, dtype=torch.bool)

        priority = build_priority_matrix(
            nodes, z=z, env=env,
            topk=builder.topk, local_window=builder.local_window,
        )

        K = max_neighbors if max_neighbors is not None else diag.max_k
        K = max(K, 1)

        nb, valid = neighbors_from_mask_prioritized(mask_t, priority, K)

        cached = CachedTopology(
            mask=mask_t,
            priority=priority,
            neighbors=nb,
            valid=valid,
            diagnostics=diag,
        )

        # Evict oldest if full
        if len(self._store) >= self.maxsize:
            oldest = self._order.pop(0)
            self._store.pop(oldest, None)

        self._store[key] = cached
        self._order.append(key)
        return cached

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def __len__(self) -> int:
        return len(self._store)
