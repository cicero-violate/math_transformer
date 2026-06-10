from __future__ import annotations

from dataclasses import dataclass

import torch

from .topology import MaskDiagnostics


@dataclass
class BlockTopologyConfig:
    block_size: int = 64
    topk_blocks: int = 4
    include_local_blocks: int = 1
    include_self_block: bool = True
    include_symbolic_bridge: bool = True
    block_token_cap: int = 16


@dataclass
class PreparedBlockTopology:
    block_neighbors: torch.Tensor      # (B, topk_blocks_eff), int64
    block_valid_i8: torch.Tensor       # (B, topk_blocks_eff), int8
    block_token_indices: torch.Tensor | None  # (B, topk_blocks_eff, C), int64 absolute token indices
    block_token_valid_i8: torch.Tensor | None # (B, topk_blocks_eff, C), int8
    token_neighbors: torch.Tensor | None      # (N, K), int64; optional for native block-only runtime
    token_valid_i8: torch.Tensor | None       # (N, K), int8; optional for native block-only runtime
    diagnostics: MaskDiagnostics
    block_size: int
    is_block_topology: bool = True

    @property
    def length(self) -> int:
        if self.token_neighbors is not None:
            return int(self.token_neighbors.shape[0])
        return int(self.diagnostics.n)

    @property
    def k(self) -> int:
        if self.token_neighbors is not None:
            return int(self.token_neighbors.shape[1])
        return 0

    @property
    def num_blocks(self) -> int:
        return int(self.block_neighbors.shape[0])

    @property
    def device(self) -> torch.device:
        if self.token_neighbors is not None:
            return self.token_neighbors.device
        return self.block_neighbors.device
