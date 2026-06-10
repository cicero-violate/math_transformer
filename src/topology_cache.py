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
    valid_i8: torch.Tensor                  # (T, K) int8
    diagnostics: MaskDiagnostics


@dataclass
class PagedNeighborTable:
    """Fixed-K neighbor table split into row pages for long-sequence storage."""

    neighbor_pages: torch.Tensor      # (P, page_size, K) long
    valid_pages: torch.Tensor         # (P, page_size, K) bool
    valid_i8_pages: torch.Tensor      # (P, page_size, K) int8
    length: int
    page_size: int
    k: int

    @property
    def num_pages(self) -> int:
        return int(self.neighbor_pages.shape[0])

    @property
    def padded_length(self) -> int:
        return int(self.num_pages * self.page_size)

    @property
    def device(self) -> torch.device:
        return self.neighbor_pages.device

    @property
    def memory_bytes(self) -> int:
        return int(
            self.neighbor_pages.numel() * self.neighbor_pages.element_size()
            + self.valid_pages.numel() * self.valid_pages.element_size()
            + self.valid_i8_pages.numel() * self.valid_i8_pages.element_size()
        )

    def page(self, page_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(neighbors, valid_i8)` for a single row page."""
        if page_idx < 0 or page_idx >= self.num_pages:
            raise IndexError(f"page_idx {page_idx} out of range for {self.num_pages} pages")
        return self.neighbor_pages[page_idx], self.valid_i8_pages[page_idx]

    def materialize_neighbors(self) -> torch.Tensor:
        return self.neighbor_pages.reshape(self.padded_length, self.k)[: self.length]

    def materialize_valid(self) -> torch.Tensor:
        return self.valid_pages.reshape(self.padded_length, self.k)[: self.length]

    def materialize_valid_i8(self) -> torch.Tensor:
        return self.valid_i8_pages.reshape(self.padded_length, self.k)[: self.length]


@dataclass
class PagedCachedTopology:
    paged_neighbors: PagedNeighborTable
    diagnostics: MaskDiagnostics
    cache_key: str


@dataclass
class PreparedTopology:
    """
    O(T*K) hot-path topology object.

    Unlike CachedTopology, this intentionally stores only tensors needed by the
    neighbor-sparse attention fast path. It avoids exposing dense mask/priority
    objects to the model block once topology has been compiled.
    """

    neighbors: torch.Tensor | None          # (T, K) long, target device; optional for native block-only
    valid_i8: torch.Tensor | None           # (T, K) int8, target device; optional for native block-only
    diagnostics: MaskDiagnostics
    cache_key: str
    block_neighbors: torch.Tensor | None = None       # (B, K_B) long, target device
    block_valid_i8: torch.Tensor | None = None        # (B, K_B) int8, target device
    block_token_indices: torch.Tensor | None = None   # (B, K_B, C) long, target device
    block_token_valid_i8: torch.Tensor | None = None  # (B, K_B, C) int8, target device
    block_size: int | None = None
    is_block_topology: bool = False

    @property
    def length(self) -> int:
        if self.neighbors is not None:
            return int(self.neighbors.shape[0])
        return int(self.diagnostics.n)

    @property
    def k(self) -> int:
        if self.neighbors is not None:
            return int(self.neighbors.shape[1])
        return 0

    @property
    def device(self) -> torch.device:
        if self.neighbors is not None:
            return self.neighbors.device
        if self.block_neighbors is not None:
            return self.block_neighbors.device
        return torch.device("cpu")

    @property
    def memory_bytes(self) -> int:
        total = 0
        if self.neighbors is not None:
            total += int(self.neighbors.numel() * self.neighbors.element_size())
        if self.valid_i8 is not None:
            total += int(self.valid_i8.numel() * self.valid_i8.element_size())
        if self.block_neighbors is not None:
            total += int(self.block_neighbors.numel() * self.block_neighbors.element_size())
        if self.block_valid_i8 is not None:
            total += int(self.block_valid_i8.numel() * self.block_valid_i8.element_size())
        if self.block_token_indices is not None:
            total += int(self.block_token_indices.numel() * self.block_token_indices.element_size())
        if self.block_token_valid_i8 is not None:
            total += int(self.block_token_valid_i8.numel() * self.block_token_valid_i8.element_size())
        return total


def page_neighbor_table(
    neighbors: torch.Tensor,
    valid: torch.Tensor,
    page_size: int = 256,
) -> PagedNeighborTable:
    """Split `(T, K)` fixed-K neighbors into padded row pages."""
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if neighbors.ndim != 2 or valid.ndim != 2:
        raise ValueError("neighbors and valid must both be rank-2 tensors")
    if neighbors.shape != valid.shape:
        raise ValueError(f"shape mismatch: neighbors={neighbors.shape}, valid={valid.shape}")

    neighbors = neighbors.long()
    valid = valid.bool()
    length = int(neighbors.shape[0])
    k = int(neighbors.shape[1])
    num_pages = max(1, (length + page_size - 1) // page_size)
    padded_length = num_pages * page_size
    pad_rows = padded_length - length

    if pad_rows:
        neighbors = torch.cat([
            neighbors,
            torch.zeros(pad_rows, k, dtype=neighbors.dtype, device=neighbors.device),
        ], dim=0)
        valid = torch.cat([
            valid,
            torch.zeros(pad_rows, k, dtype=valid.dtype, device=valid.device),
        ], dim=0)

    neighbor_pages = neighbors.reshape(num_pages, page_size, k).contiguous()
    valid_pages = valid.reshape(num_pages, page_size, k).contiguous()
    valid_i8_pages = valid_pages.char().contiguous()
    return PagedNeighborTable(
        neighbor_pages=neighbor_pages,
        valid_pages=valid_pages,
        valid_i8_pages=valid_i8_pages,
        length=length,
        page_size=page_size,
        k=k,
    )


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
    topology_mode: str = "union",
    fixed_k: int | None = None,
    middle_bridge_width: int = 0,
) -> str:
    return "|".join([
        stable_nodes_hash(nodes),
        stable_env_hash(env),
        str(topk),
        str(local_window),
        str(max_neighbors),
        str(topology_mode),
        str(fixed_k),
        str(middle_bridge_width),
        device,
    ])


class TopologyCache:
    """
    In-process LRU-style cache for topology builds and node embeddings.

    Topology keyed on (nodes_hash, env_hash, topk, local_window, max_neighbors, device).
    Embeddings keyed on nodes_hash — encode_batch is called at most once per unique node set.
    """

    def __init__(self, maxsize: int = 256) -> None:
        self._store: dict[str, CachedTopology] = {}
        self._paged_store: dict[str, PagedCachedTopology] = {}
        self._prepared_store: dict[str, PreparedTopology] = {}
        self._order: list[str] = []   # insertion-order eviction
        self._z_store: dict[str, np.ndarray] = {}  # Sprint 2: frozen embeddings
        self.maxsize = maxsize
        self.cache_hits = 0
        self.cache_misses = 0

    def get_or_encode(self, nodes: list[MathNode], embedder) -> np.ndarray:
        """Return cached embedding array; compute and cache on first call."""
        key = stable_nodes_hash(nodes)
        if key not in self._z_store:
            self._z_store[key] = embedder.encode_batch(nodes)
        return self._z_store[key]

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
        mode = getattr(builder, "topology_mode", "union")
        include_middle_bridge = bool(getattr(builder, "include_middle_bridge", False))
        middle_bridge_width = int(getattr(builder, "middle_bridge_width", 0))
        key = _cache_key(
            nodes, env, builder.topk, builder.local_window, max_neighbors, str(dev),
            topology_mode=mode,
            fixed_k=getattr(builder, "fixed_k", None),
            middle_bridge_width=middle_bridge_width,
        )

        if key in self._store:
            self.cache_hits += 1
            return self._store[key]

        self.cache_misses += 1

        if use_gpu:
            Z_t = torch.tensor(z, dtype=torch.float32, device=dev)
            if bool(getattr(builder, "is_learned_topology", False)):
                mask_t, diag = builder.build_scored_topk_torch(nodes, Z_t, env, dev)
                priority_t = builder.priority_from_mask_torch(mask_t)
            elif mode in ("scored_topk", "middle_preserving_topk"):
                mask_t, diag = builder.build_scored_topk_torch(nodes, Z_t, env, dev)
                priority_t = build_priority_matrix_torch(
                    nodes, Z_t=Z_t, env=env,
                    topk=builder.topk, local_window=builder.local_window, device=dev,
                    include_middle_bridge=include_middle_bridge,
                    middle_bridge_width=middle_bridge_width,
                )
            else:
                mask_t, diag = builder.build_detailed_torch(nodes, Z_t, env, dev)
                priority_t = build_priority_matrix_torch(
                    nodes, Z_t=Z_t, env=env,
                    topk=builder.topk, local_window=builder.local_window, device=dev,
                    include_middle_bridge=include_middle_bridge,
                    middle_bridge_width=middle_bridge_width,
                )
            K = max(max_neighbors if max_neighbors is not None else diag.max_k, 1)
            nb, valid = neighbors_from_priority_torch(priority_t, K)
            cached = CachedTopology(
                mask=mask_t,
                priority=priority_t,
                neighbors=nb,
                valid=valid,
                valid_i8=valid.char(),
                diagnostics=diag,
            )
        else:
            if bool(getattr(builder, "is_learned_topology", False)):
                np_mask, diag = builder.build_scored_topk(nodes, z, env)
                priority = builder.priority_from_mask(np_mask)
            elif mode in ("scored_topk", "middle_preserving_topk"):
                np_mask, diag = builder.build_scored_topk(nodes, z, env)
                priority = build_priority_matrix(
                    nodes, z=z, env=env,
                    topk=builder.topk, local_window=builder.local_window,
                    include_middle_bridge=include_middle_bridge,
                    middle_bridge_width=middle_bridge_width,
                )
            else:
                np_mask, diag = builder.build_detailed(nodes, z, env)
                priority = build_priority_matrix(
                    nodes, z=z, env=env,
                    topk=builder.topk, local_window=builder.local_window,
                    include_middle_bridge=include_middle_bridge,
                    middle_bridge_width=middle_bridge_width,
                )
            mask_t = torch.tensor(np_mask, dtype=torch.bool)
            K = max(max_neighbors if max_neighbors is not None else diag.max_k, 1)
            nb, valid = neighbors_from_mask_prioritized(mask_t, priority, K)
            cached = CachedTopology(
                mask=mask_t,
                priority=priority,
                neighbors=nb,
                valid=valid,
                valid_i8=valid.char(),
                diagnostics=diag,
            )

        if len(self._store) >= self.maxsize:
            oldest = self._order.pop(0)
            self._store.pop(oldest, None)

        self._store[key] = cached
        self._order.append(key)
        return cached

    def get_or_prepare(
        self,
        nodes: list[MathNode],
        z: np.ndarray,
        env: dict | None,
        builder,
        max_neighbors: int | None = None,
        device: torch.device | None = None,
    ) -> PreparedTopology:
        """Return an O(T*K) prepared topology for the fast path."""
        dev = device if device is not None else torch.device("cpu")
        mode = getattr(builder, "topology_mode", "union")
        prepare_mode = getattr(builder, "prepare_mode", "full")
        middle_bridge_width = int(getattr(builder, "middle_bridge_width", 0))
        key = _cache_key(
            nodes, env, builder.topk, builder.local_window, max_neighbors, str(dev),
            topology_mode=mode,
            fixed_k=getattr(builder, "fixed_k", None),
            middle_bridge_width=middle_bridge_width,
        ) + "|" + str(getattr(builder, "cache_config_key", "")) + f"|prepare_mode={prepare_mode}|prepared"

        if key in self._prepared_store:
            self.cache_hits += 1
            return self._prepared_store[key]

        if hasattr(builder, "prepare_topology"):
            prepared = builder.prepare_topology(
                nodes, z, env, max_neighbors=max_neighbors, device=dev
            )
            block_neighbors = getattr(prepared, "block_neighbors", None)
            block_valid_i8 = getattr(prepared, "block_valid_i8", None)
            block_token_indices = getattr(prepared, "block_token_indices", None)
            block_token_valid_i8 = getattr(prepared, "block_token_valid_i8", None)
            out = PreparedTopology(
                neighbors=None if prepared.token_neighbors is None else prepared.token_neighbors.to(dev).contiguous(),
                valid_i8=None if prepared.token_valid_i8 is None else prepared.token_valid_i8.to(dev).contiguous(),
                diagnostics=prepared.diagnostics,
                cache_key=key,
                block_neighbors=None if block_neighbors is None else block_neighbors.to(dev).contiguous(),
                block_valid_i8=None if block_valid_i8 is None else block_valid_i8.to(dev).contiguous(),
                block_token_indices=None if block_token_indices is None else block_token_indices.to(dev).contiguous(),
                block_token_valid_i8=None if block_token_valid_i8 is None else block_token_valid_i8.to(dev).contiguous(),
                block_size=getattr(prepared, "block_size", None),
                is_block_topology=bool(getattr(prepared, "is_block_topology", False)),
            )
            self._prepared_store[key] = out
            return out

        cached = self.get_or_build(
            nodes, z, env, builder,
            max_neighbors=max_neighbors,
            device=dev,
        )
        out = PreparedTopology(
            neighbors=cached.neighbors.to(dev).contiguous(),
            valid_i8=cached.valid_i8.to(dev).contiguous(),
            diagnostics=cached.diagnostics,
            cache_key=key,
        )
        self._prepared_store[key] = out
        return out

    def get_or_build_paged(
        self,
        nodes: list[MathNode],
        z: np.ndarray,
        env: dict | None,
        builder,
        max_neighbors: int | None = None,
        device: torch.device | None = None,
        page_size: int = 256,
    ) -> PagedCachedTopology:
        """
        Return fixed-K neighbors as row pages.

        The current topology builders may still create dense masks/priorities
        transiently. The persisted paged cache object stores only O(T*K)
        neighbor and validity tensors, which is the representation needed for
        long-sequence page-wise GPU execution.
        """
        dev = device if device is not None else torch.device("cpu")
        mode = getattr(builder, "topology_mode", "union")
        middle_bridge_width = int(getattr(builder, "middle_bridge_width", 0))
        key = _cache_key(
            nodes, env, builder.topk, builder.local_window, max_neighbors, str(dev),
            topology_mode=mode,
            fixed_k=getattr(builder, "fixed_k", None),
            middle_bridge_width=middle_bridge_width,
        ) + f"|paged|page_size={page_size}"

        if key in self._paged_store:
            self.cache_hits += 1
            return self._paged_store[key]

        cached = self.get_or_build(
            nodes, z, env, builder,
            max_neighbors=max_neighbors,
            device=dev,
        )
        paged = page_neighbor_table(
            cached.neighbors.to(dev),
            cached.valid.to(dev),
            page_size=page_size,
        )
        out = PagedCachedTopology(
            paged_neighbors=paged,
            diagnostics=cached.diagnostics,
            cache_key=key,
        )
        self._paged_store[key] = out
        return out

    def clear(self) -> None:
        self._store.clear()
        self._paged_store.clear()
        self._prepared_store.clear()
        self._order.clear()
        self._z_store.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def __len__(self) -> int:
        return len(self._store)
