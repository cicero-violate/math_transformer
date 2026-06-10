from __future__ import annotations
from typing import Literal
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ir import MathNode
from .embedder import MathEmbedder, EMBED_DIM
from .topology import TopologyBuilder, MaskDiagnostics
from .topology_cache import TopologyCache, CachedTopology, PreparedTopology
from .attention import DenseMaskedMathAttention, NeighborSparseMathAttention, BlockSparseMathAttention, SparseSelectorMode
from .sparse_attention import symbolic_priority_scores
from .router import OperatorRouter, RouteResult

AttentionMode = Literal["full", "dense_masked", "neighbor_sparse", "block_sparse"]
DEFAULT_MAX_NEIGHBORS = 16


class MathRoutedTransformerBlock(nn.Module):
    """
    Single transformer block with three attention modes:
      full            — no mask, dense QK^T
      dense_masked    — topology mask + dense QK^T  (correctness baseline)
      neighbor_sparse — topology mask → priority neighbor list → O(T·K·D)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        topk: int = 4,
        local_window: int = 2,
        attention_mode: AttentionMode = "dense_masked",
        max_neighbors: int | None = DEFAULT_MAX_NEIGHBORS,
        topology_cache: TopologyCache | None = None,
        topology_mode: str = "union",
        fixed_k: int = 32,
        relation_weights: dict | None = None,
        middle_bridge_width: int = 0,
        sparse_selector: SparseSelectorMode = "topology_only",
        selector_alpha: float = 1.0,
        selector_beta: float = 1.0,
        selector_k: int | None = None,
        triton_block_d: int | None = None,
        triton_block_k: int | None = None,
    ) -> None:
        super().__init__()
        self.attention_mode = attention_mode
        self.max_neighbors = max_neighbors
        self._topology_cache = topology_cache
        self.sparse_selector = sparse_selector
        self.selector_alpha = selector_alpha
        self.selector_beta = selector_beta
        self.selector_k = selector_k
        self._prepared_topology: PreparedTopology | None = None
        self.register_buffer("static_neighbors", torch.empty(0, 0, dtype=torch.long), persistent=False)
        self.register_buffer("static_valid_i8", torch.empty(0, 0, dtype=torch.int8), persistent=False)
        self.register_buffer("static_block_neighbors", torch.empty(0, 0, dtype=torch.long), persistent=False)
        self.register_buffer("static_block_valid_i8", torch.empty(0, 0, dtype=torch.int8), persistent=False)
        self.register_buffer("static_block_token_indices", torch.empty(0, 0, 0, dtype=torch.long), persistent=False)
        self.register_buffer("static_block_token_valid_i8", torch.empty(0, 0, 0, dtype=torch.int8), persistent=False)
        self.static_block_size: int | None = None

        if attention_mode == "neighbor_sparse":
            self.attn = NeighborSparseMathAttention(
                d_model, n_heads, dropout,
                triton_block_d=triton_block_d,
                triton_block_k=triton_block_k,
            )
        elif attention_mode == "block_sparse":
            self.attn = BlockSparseMathAttention(d_model, n_heads, dropout)
        else:
            self.attn = DenseMaskedMathAttention(d_model, n_heads, dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.topology = TopologyBuilder(
            topk=topk, local_window=local_window,
            topology_mode=topology_mode, fixed_k=fixed_k,
            relation_weights=relation_weights,
            middle_bridge_width=middle_bridge_width,
        )
        self.embedder = MathEmbedder()
        self.router = OperatorRouter()

    def prepare_static_topology(
        self,
        nodes: list[MathNode],
        env: dict[str, tuple[int, ...]] | None = None,
        device: torch.device | None = None,
    ) -> PreparedTopology:
        """Compile and store O(T*K) neighbor tensors for repeated fast-path use."""
        dev = device if device is not None else self.static_neighbors.device
        if dev.type == "meta":
            dev = torch.device("cpu")
        cache = self._topology_cache if self._topology_cache is not None else TopologyCache(maxsize=1)
        z = cache.get_or_encode(nodes, self.embedder)
        prepared = cache.get_or_prepare(
            nodes, z, env, self.topology, self.max_neighbors, device=dev
        )
        self._prepared_topology = prepared
        self.static_neighbors = prepared.neighbors.to(dev).contiguous()
        self.static_valid_i8 = prepared.valid_i8.to(dev).contiguous()
        if prepared.block_neighbors is not None:
            self.static_block_neighbors = prepared.block_neighbors.to(dev).contiguous()
        else:
            self.static_block_neighbors = torch.empty(0, 0, dtype=torch.long, device=dev)
        if prepared.block_valid_i8 is not None:
            self.static_block_valid_i8 = prepared.block_valid_i8.to(dev).contiguous()
        else:
            self.static_block_valid_i8 = torch.empty(0, 0, dtype=torch.int8, device=dev)
        if prepared.block_token_indices is not None:
            self.static_block_token_indices = prepared.block_token_indices.to(dev).contiguous()
        else:
            self.static_block_token_indices = torch.empty(0, 0, 0, dtype=torch.long, device=dev)
        if prepared.block_token_valid_i8 is not None:
            self.static_block_token_valid_i8 = prepared.block_token_valid_i8.to(dev).contiguous()
        else:
            self.static_block_token_valid_i8 = torch.empty(0, 0, 0, dtype=torch.int8, device=dev)
        self.static_block_size = prepared.block_size
        return prepared

    def _prepared_or_cached_topology(
        self,
        x: torch.Tensor,
        nodes: list[MathNode],
        env: dict[str, tuple[int, ...]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, PreparedTopology | None]:
        if (
            self.static_neighbors.numel() > 0
            and self.static_valid_i8.numel() > 0
            and self.static_neighbors.shape[0] == len(nodes)
            and self.static_neighbors.device == x.device
        ):
            return self.static_neighbors, self.static_valid_i8, self._prepared_topology

        cache = self._topology_cache if self._topology_cache is not None else TopologyCache(maxsize=1)
        z = cache.get_or_encode(nodes, self.embedder)
        prepared = cache.get_or_prepare(
            nodes, z, env, self.topology, self.max_neighbors, device=x.device
        )
        return prepared.neighbors, prepared.valid_i8, prepared

    def _ff_inference(self, x: torch.Tensor) -> torch.Tensor:
        x = F.linear(x, self.ff[0].weight, self.ff[0].bias)
        x = F.gelu(x)
        return F.linear(x, self.ff[3].weight, self.ff[3].bias)

    def _maybe_residual_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.drop.p > 0.0:
            return self.drop(x)
        return x

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.ff(x)
        return self._ff_inference(x)

    def forward_cached_fast_path(
        self,
        x: torch.Tensor,
        nodes: list[MathNode],
        env: dict[str, tuple[int, ...]] | None = None,
    ) -> torch.Tensor:
        nb, valid_i8, _prepared = self._prepared_or_cached_topology(x, nodes, env)
        if self.attention_mode == "block_sparse":
            if _prepared is None or not _prepared.is_block_topology or _prepared.block_neighbors is None or _prepared.block_size is None:
                raise RuntimeError("block_sparse attention requires prepared block topology")
            attn_out = self.attn(
                self.norm1(x),
                _prepared.block_neighbors,
                _prepared.block_valid_i8,
                _prepared.block_size,
                block_token_indices=_prepared.block_token_indices,
                block_token_valid_i8=_prepared.block_token_valid_i8,
            )
        else:
            symbolic_scores = None
            attn_out = self.attn.forward_fused_norm_qkv(
                x, self.norm1, nb, valid_i8,
                selector_mode=self.sparse_selector,
                symbolic_scores=symbolic_scores,
                selector_alpha=self.selector_alpha,
                selector_beta=self.selector_beta,
                selector_k=self.selector_k,
            )
        x = x.add(self._maybe_residual_dropout(attn_out))
        return x.add(self._maybe_residual_dropout(self._ff_block(self.norm2(x))))

    def forward_static_fast_path(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference using already-prepared static topology buffers."""
        if self.attention_mode not in ("neighbor_sparse", "block_sparse"):
            raise RuntimeError("forward_static_fast_path requires sparse attention")
        if self.static_neighbors.numel() == 0 or self.static_valid_i8.numel() == 0:
            raise RuntimeError("forward_static_fast_path requires prepare_static_topology first")
        if self.static_neighbors.shape[0] != x.shape[1]:
            raise RuntimeError("static topology length does not match input sequence length")
        if self.static_neighbors.device != x.device:
            raise RuntimeError("static topology device does not match input device")
        if self.attention_mode == "block_sparse":
            if self.static_block_neighbors.numel() == 0 or self.static_block_valid_i8.numel() == 0 or self.static_block_size is None:
                raise RuntimeError("block_sparse static path requires block topology fields")
            attn_out = self.attn(
                self.norm1(x),
                self.static_block_neighbors,
                self.static_block_valid_i8,
                self.static_block_size,
                block_token_indices=self.static_block_token_indices if self.static_block_token_indices.numel() else None,
                block_token_valid_i8=self.static_block_token_valid_i8 if self.static_block_token_valid_i8.numel() else None,
            )
        else:
            attn_out = self.attn.forward_fused_norm_qkv(
                x, self.norm1, self.static_neighbors, self.static_valid_i8,
                selector_mode=self.sparse_selector,
                symbolic_scores=None,
                selector_alpha=self.selector_alpha,
                selector_beta=self.selector_beta,
                selector_k=self.selector_k,
            )
        x = x.add(self._maybe_residual_dropout(attn_out))
        return x.add(self._maybe_residual_dropout(self._ff_block(self.norm2(x))))

    def forward(
        self,
        x: torch.Tensor,
        nodes: list[MathNode] | None = None,
        env: dict[str, tuple[int, ...]] | None = None,
        return_metadata: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[RouteResult] | None, MaskDiagnostics | None]:
        """
        x     : (B, T, d_model)
        nodes : list[MathNode] of length T, or None for full attention
        env   : shape/type environment for shape_compat and composition relations
        return_metadata=False skips router/diagnostic return work on cache hits.
        Returns (output, mask, route_info, diagnostics)
        """
        B, T, _ = x.shape
        if (
            self.attention_mode == "block_sparse"
            and not return_metadata
            and nodes is not None
            and len(nodes) == T
        ):
            return self.forward_cached_fast_path(x, nodes, env), None, None, None
        if (
            self.attention_mode == "neighbor_sparse"
            and not return_metadata
            and nodes is not None
            and len(nodes) == T
        ):
            return self.forward_cached_fast_path(x, nodes, env), None, None, None

        mask: torch.Tensor | None = None
        route_info: list[RouteResult] | None = None
        diag: MaskDiagnostics | None = None

        if self.attention_mode == "block_sparse" and nodes is not None and len(nodes) == T:
            cache = self._topology_cache if self._topology_cache is not None else TopologyCache(maxsize=1)
            z = cache.get_or_encode(nodes, self.embedder)
            prepared = cache.get_or_prepare(
                nodes, z, env, self.topology, self.max_neighbors, device=x.device
            )
            if not prepared.is_block_topology or prepared.block_neighbors is None or prepared.block_size is None:
                raise RuntimeError("block_sparse attention requires a block topology builder")
            diag = prepared.diagnostics if return_metadata else None
            route_info = self.router.route_batch(nodes, z) if return_metadata else None
            attn_out = self.attn(
                self.norm1(x),
                prepared.block_neighbors,
                prepared.block_valid_i8,
                prepared.block_size,
                block_token_indices=prepared.block_token_indices,
                block_token_valid_i8=prepared.block_token_valid_i8,
            )
            x = x + self._maybe_residual_dropout(attn_out)
            x = x + self._maybe_residual_dropout(self._ff_block(self.norm2(x)))
            return x, None, route_info, diag

        if nodes is not None and len(nodes) == T and self.attention_mode != "full":
            # Do not use `or` here: TopologyCache implements __len__, so an
            # empty shared cache is falsy and would be discarded before warming.
            cache = self._topology_cache if self._topology_cache is not None else TopologyCache(maxsize=1)
            z = cache.get_or_encode(nodes, self.embedder)

            if self.attention_mode == "neighbor_sparse":
                # Use cache path → builds priority + prioritized neighbors
                cached: CachedTopology = cache.get_or_build(
                    nodes, z, env, self.topology, self.max_neighbors,
                    device=x.device,
                )
                mask = cached.mask.to(x.device)
                nb = cached.neighbors.to(x.device)
                valid = cached.valid_i8.to(x.device)
                symbolic_scores = (
                    symbolic_priority_scores(torch.as_tensor(cached.priority, device=x.device))
                    if self.sparse_selector in ("symbolic_kmip", "symbolic_candidate_kmip")
                    else None
                )
                diag = cached.diagnostics
            else:
                np_mask, diag = self.topology.build_detailed(nodes, z, env)
                mask = torch.tensor(np_mask, dtype=torch.bool, device=x.device)

            if return_metadata:
                route_info = self.router.route_batch(nodes, z)
            else:
                mask = None
                diag = None

        x_normed = self.norm1(x)

        if self.attention_mode == "neighbor_sparse" and nodes is not None and len(nodes) == T:
            attn_out = self.attn(
                x_normed, nb, valid,
                selector_mode=self.sparse_selector,
                symbolic_scores=symbolic_scores,
                selector_alpha=self.selector_alpha,
                selector_beta=self.selector_beta,
                selector_k=self.selector_k,
            )
        elif self.attention_mode == "full":
            attn_out = self.attn(x_normed, mask=None)
        else:
            attn_out = self.attn(x_normed, mask=mask)

        x = x + self._maybe_residual_dropout(attn_out)
        x = x + self._maybe_residual_dropout(self._ff_block(self.norm2(x)))
        return x, mask, route_info, diag

    def profile_cached_sparse_block(
        self,
        x: torch.Tensor,
        nodes: list[MathNode],
        env: dict[str, tuple[int, ...]] | None = None,
        fused_norm_qkv: bool = False,
        fused_attn_outproj: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Profile the topology-only sparse block with coarse timing buckets.

        The optional fused flags keep the prepared benchmark auditable by
        replacing separate bucket pairs with the corresponding fused bucket
        when the CUDA/Triton fast path is available.
        """
        if self.attention_mode not in ("neighbor_sparse", "block_sparse"):
            raise RuntimeError("profile_cached_sparse_block requires sparse attention")
        if self.sparse_selector != "topology_only":
            raise RuntimeError("profile_cached_sparse_block currently supports topology_only only")

        def sync() -> None:
            if x.device.type == "cuda":
                torch.cuda.synchronize(x.device)

        def mark(prev: float, key: str, out: dict[str, float]) -> float:
            sync()
            now = time.perf_counter()
            out[key] = (now - prev) * 1000.0
            return now

        timings: dict[str, float] = {}
        sync()
        t = time.perf_counter()
        nb, valid_i8, prepared = self._prepared_or_cached_topology(x, nodes, env)
        t = mark(t, "topology_prepare_ms", timings)

        from .triton_attention import triton_neighbor_attention_flat, TRITON_AVAILABLE
        from .sparse_attention import neighbor_attention

        use_fused_norm_qkv = bool(fused_norm_qkv and TRITON_AVAILABLE and x.is_cuda and not self.training)
        use_fused_attn_outproj = bool(fused_attn_outproj and TRITON_AVAILABLE and x.is_cuda and not self.training)

        if use_fused_norm_qkv:
            q, k, v = self.attn.project_norm_qkv_for_profile(x, self.norm1)
            t = mark(t, "norm_qkv_ms", timings)
        else:
            x1 = self.norm1(x)
            t = mark(t, "norm1_ms", timings)
            q, k, v = self.attn.project_qkv_for_profile(x1)
            t = mark(t, "qkv_ms", timings)

        if self.attention_mode == "block_sparse":
            if prepared is None or not prepared.is_block_topology or prepared.block_neighbors is None or prepared.block_size is None:
                raise RuntimeError("block_sparse profiling requires prepared block topology")
            if use_fused_attn_outproj:
                attn_out = self.attn.attention_out_project_for_profile(
                    q, k, v, prepared.block_neighbors, prepared.block_valid_i8, prepared.block_size,
                    block_token_indices=prepared.block_token_indices,
                    block_token_valid_i8=prepared.block_token_valid_i8,
                )
                try:
                    from .block_sparse_attention import block_sparse_attention_last_backend
                    backend = block_sparse_attention_last_backend()
                except Exception:
                    backend = "unknown"
                timings["native_block_backend_triton"] = 1.0 if backend == "triton" else 0.0
                timings["native_block_backend_vectorized"] = 1.0 if backend == "vectorized" else 0.0
                t = mark(t, "attention_outproj_ms", timings)
            else:
                attn_raw_bhtd = self.attn.block_attention_for_profile(
                    q, k, v, prepared.block_neighbors, prepared.block_valid_i8, prepared.block_size,
                    block_token_indices=prepared.block_token_indices,
                    block_token_valid_i8=prepared.block_token_valid_i8,
                )
                try:
                    from .block_sparse_attention import block_sparse_attention_last_backend
                    backend = block_sparse_attention_last_backend()
                except Exception:
                    backend = "unknown"
                timings["native_block_backend_triton"] = 1.0 if backend == "triton" else 0.0
                timings["native_block_backend_vectorized"] = 1.0 if backend == "vectorized" else 0.0
                t = mark(t, "attention_kernel_ms", timings)
                Bp, Hp, Tp, Dp = attn_raw_bhtd.shape
                attn_raw = attn_raw_bhtd.transpose(1, 2).contiguous().reshape(Bp, Tp, Hp * Dp)
                attn_out = self.attn.out_project_for_profile(attn_raw)
                t = mark(t, "out_proj_ms", timings)
        elif use_fused_attn_outproj:
            attn_out = self.attn.attention_out_project_for_profile(q, k, v, nb, valid_i8)
            t = mark(t, "attention_outproj_ms", timings)
        else:
            if TRITON_AVAILABLE and x.is_cuda:
                attn_raw = triton_neighbor_attention_flat(
                    q, k, v, nb, valid_i8,
                    block_d=self.attn.triton_block_d,
                    block_k=self.attn.triton_block_k,
                )
            else:
                attn_raw = neighbor_attention(q, k, v, nb, valid_i8.bool())
                B, T, _ = x.shape
                attn_raw = attn_raw.transpose(1, 2).reshape(B, T, self.attn.d_model)
            t = mark(t, "attention_kernel_ms", timings)
            attn_out = self.attn.out_project_for_profile(attn_raw)
            t = mark(t, "out_proj_ms", timings)
        y = x.add(self._maybe_residual_dropout(attn_out))
        t = mark(t, "residual1_ms", timings)
        y2 = self.norm2(y)
        t = mark(t, "norm2_ms", timings)
        ff = self._ff_block(y2)
        t = mark(t, "ffn_ms", timings)
        out = y.add(self._maybe_residual_dropout(ff))
        t = mark(t, "residual2_ms", timings)
        duration_keys = (
            "topology_prepare_ms",
            "norm1_ms",
            "qkv_ms",
            "norm_qkv_ms",
            "attention_kernel_ms",
            "out_proj_ms",
            "attention_outproj_ms",
            "residual1_ms",
            "norm2_ms",
            "ffn_ms",
            "residual2_ms",
        )
        timings["total_block_ms"] = sum(float(timings.get(k, 0.0)) for k in duration_keys)
        return out, timings


class MathRoutedTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        topk: int = 4,
        local_window: int = 2,
        n_experts: int = 7,
        attention_mode: AttentionMode = "dense_masked",
        max_neighbors: int | None = DEFAULT_MAX_NEIGHBORS,
        share_topology_cache: bool = True,
        topology_mode: str = "union",
        fixed_k: int = 32,
        relation_weights: dict | None = None,
        middle_bridge_width: int = 0,
        sparse_selector: SparseSelectorMode = "topology_only",
        selector_alpha: float = 1.0,
        selector_beta: float = 1.0,
        selector_k: int | None = None,
        triton_block_d: int | None = None,
        triton_block_k: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.attention_mode = attention_mode

        shared_cache = TopologyCache() if share_topology_cache else None

        self.embed_proj = nn.Linear(EMBED_DIM, d_model)
        self.layers = nn.ModuleList([
            MathRoutedTransformerBlock(
                d_model, n_heads, d_ff, dropout, topk, local_window,
                attention_mode, max_neighbors,
                topology_cache=shared_cache,
                topology_mode=topology_mode,
                fixed_k=fixed_k,
                relation_weights=relation_weights,
                middle_bridge_width=middle_bridge_width,
                sparse_selector=sparse_selector,
                selector_alpha=selector_alpha,
                selector_beta=selector_beta,
                selector_k=selector_k,
                triton_block_d=triton_block_d,
                triton_block_k=triton_block_k,
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, d_model)
        self.route_head = nn.Linear(d_model, n_experts)
        self._embedder = MathEmbedder()

    def prepare_static_topology(
        self,
        nodes: list[MathNode],
        env: dict[str, tuple[int, ...]] | None = None,
        device: torch.device | None = None,
    ) -> list[PreparedTopology]:
        dev = device if device is not None else next(self.parameters()).device
        prepared: list[PreparedTopology] = []
        for layer in self.layers:
            prepared.append(layer.prepare_static_topology(nodes, env=env, device=dev))
        return prepared

    def forward(
        self,
        x: torch.Tensor,
        nodes: list[MathNode] | None = None,
        env: dict[str, tuple[int, ...]] | None = None,
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, list, list] | tuple[torch.Tensor, list, list, list]:
        masks, routes, diags = [], [], []
        for layer in self.layers:
            x, mask, route_info, diag = layer(
                x, nodes, env=env, return_metadata=True
            )
            masks.append(mask)
            routes.append(route_info)
            diags.append(diag)
        out = self.head(x)
        if return_diagnostics:
            return out, masks, routes, diags
        return out, masks, routes

    def forward_static_fast_path(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transformer using precompiled static sparse topology buffers."""
        for layer in self.layers:
            x = layer.forward_static_fast_path(x)
        return self.head(x)

    def route_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Predict expert logits from root token. x: (B, T, d)."""
        return self.route_head(x[:, 0, :])

    def embed_nodes(self, nodes: list[MathNode]) -> torch.Tensor:
        """Convert MathNodes to initial embeddings: (1, T, d_model)."""
        z = self._embedder.encode_batch(nodes)
        device = next(self.parameters()).device
        z_t = torch.tensor(z, dtype=torch.float32, device=device)
        return self.embed_proj(z_t).unsqueeze(0)
