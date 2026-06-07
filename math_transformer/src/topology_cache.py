from __future__ import annotations
import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from .ir import MathNode
from .topology import MaskDiagnostics


@dataclass
class CachedTopology:
    mask: torch.Tensor                      # (T, T) bool
    priority: np.ndarray | torch.Tensor    # (T, T) int8
    neighbors: torch.Tensor                 # (T, K) long
    valid: torch.Tensor                     # (T, K) bool
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
    device: str = "cpu",
) -> str:
    return "|".join([
        stable_nodes_hash(nodes),
        stable_env_hash(env),
        str(topk),
        str(local_window),
        str(max_neighbors),
        device,
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
        builder,                          # TopologyBuilder instance
        max_neighbors: int | None = None,
        device: torch.device | None = None,
    ) -> CachedTopology:
        from .topology import build_priority_matrix, build_priority_matrix_torch
        from .sparse_attention import neighbors_from_mask_prioritized, neighbors_from_priority_torch

        dev = device if device is not None else torch.device("cpu")
        use_gpu = dev.type != "cpu"
        key = _cache_key(nodes, env, builder.topk, builder.local_window, max_neighbors, str(dev))

        if key in self._store:
            self.cache_hits += 1
            return self._store[key]

        self.cache_misses += 1

        if use_gpu:
            Z_t = torch.tensor(z, dtype=torch.float32, device=dev)
            mask_t, diag = builder.build_detailed_torch(nodes, Z_t, env, dev)
            priority_t = build_priority_matrix_torch(
                nodes, Z_t=Z_t, env=env,
                topk=builder.topk, local_window=builder.local_window, device=dev,
            )
            K = max(max_neighbors if max_neighbors is not None else diag.max_k, 1)
            nb, valid = neighbors_from_priority_torch(priority_t, K)
            cached = CachedTopology(
                mask=mask_t,
                priority=priority_t,
                neighbors=nb,
                valid=valid,
                diagnostics=diag,
            )
        else:
            np_mask, diag = builder.build_detailed(nodes, z, env)
            mask_t = torch.tensor(np_mask, dtype=torch.bool)
            priority = build_priority_matrix(
                nodes, z=z, env=env,
                topk=builder.topk, local_window=builder.local_window,
            )
            K = max(max_neighbors if max_neighbors is not None else diag.max_k, 1)
            nb, valid = neighbors_from_mask_prioritized(mask_t, priority, K)
            cached = CachedTopology(
                mask=mask_t,
                priority=priority,
                neighbors=nb,
                valid=valid,
                diagnostics=diag,
            )

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
