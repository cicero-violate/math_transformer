from __future__ import annotations
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from .ir import MathNode
from .embedder import MathEmbedder, EMBED_DIM
from .topology import TopologyBuilder, MaskDiagnostics
from .topology_cache import TopologyCache, CachedTopology
from .attention import DenseMaskedMathAttention, NeighborSparseMathAttention
from .router import OperatorRouter, RouteResult

AttentionMode = Literal["full", "dense_masked", "neighbor_sparse"]


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
        max_neighbors: int | None = None,
        topology_cache: TopologyCache | None = None,
        topology_mode: str = "union",
        fixed_k: int = 32,
        relation_weights: dict | None = None,
    ) -> None:
        super().__init__()
        self.attention_mode = attention_mode
        self.max_neighbors = max_neighbors
        self._topology_cache = topology_cache

        if attention_mode == "neighbor_sparse":
            self.attn = NeighborSparseMathAttention(d_model, n_heads, dropout)
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
        )
        self.embedder = MathEmbedder()
        self.router = OperatorRouter()

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
        mask: torch.Tensor | None = None
        route_info: list[RouteResult] | None = None
        diag: MaskDiagnostics | None = None

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
                valid = cached.valid.to(x.device)
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
            attn_out = self.attn(x_normed, nb, valid)
        elif self.attention_mode == "full":
            attn_out = self.attn(x_normed, mask=None)
        else:
            attn_out = self.attn(x_normed, mask=mask)

        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, mask, route_info, diag


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
        max_neighbors: int | None = None,
        share_topology_cache: bool = True,
        topology_mode: str = "union",
        fixed_k: int = 32,
        relation_weights: dict | None = None,
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
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, d_model)
        self.route_head = nn.Linear(d_model, n_experts)
        self._embedder = MathEmbedder()

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

    def route_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Predict expert logits from root token. x: (B, T, d)."""
        return self.route_head(x[:, 0, :])

    def embed_nodes(self, nodes: list[MathNode]) -> torch.Tensor:
        """Convert MathNodes to initial embeddings: (1, T, d_model)."""
        z = self._embedder.encode_batch(nodes)
        device = next(self.parameters()).device
        z_t = torch.tensor(z, dtype=torch.float32, device=device)
        return self.embed_proj(z_t).unsqueeze(0)
