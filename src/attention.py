from __future__ import annotations
import math
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Core attention functions ──────────────────────────────────────────────────

def math_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Dense masked scaled dot-product attention (correctness baseline).
    q, k, v : (..., T, D)
    mask    : (..., T, T) bool — True = allowed to attend
    """
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = F.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return probs @ v


# ── Projection mixin ──────────────────────────────────────────────────────────

class _AttentionBase(nn.Module):
    """Shared Q/K/V + output projections for all attention variants."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_head
        def _p(proj: nn.Linear) -> torch.Tensor:
            return proj(x).view(B, T, H, Dh).transpose(1, 2)
        return _p(self.q_proj), _p(self.k_proj), _p(self.v_proj)

    def _maybe_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.dropout.p > 0.0:
            return self.dropout(x)
        return x

    def _collect(self, out: torch.Tensor, B: int, T: int) -> torch.Tensor:
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(self._maybe_dropout(out))

    def _project_bhtd(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H, Dh = self.n_heads, self.d_head
        def _p(proj: nn.Linear) -> torch.Tensor:
            w = proj.weight.view(H, Dh, self.d_model)
            return torch.einsum("btd,hfd->bhtf", x, w)
        return _p(self.q_proj), _p(self.k_proj), _p(self.v_proj)

    def _collect_bhtd(self, out: torch.Tensor) -> torch.Tensor:
        H, Dh = self.n_heads, self.d_head
        w = self.out_proj.weight.view(self.d_model, H, Dh)
        y = torch.einsum("bhtd,ohd->bto", self._maybe_dropout(out), w)
        if self.out_proj.bias is not None:
            y = y + self.out_proj.bias
        return y


# ── Dense masked attention ────────────────────────────────────────────────────

class DenseMaskedMathAttention(_AttentionBase):
    """
    Multi-head attention with an external boolean topology mask.
    Dense O(T² D) — correctness reference implementation.
    """

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        x    : (B, T, d_model)
        mask : (T, T) or (B, T, T) bool — True means attend
        """
        B, T, _ = x.shape
        q, k, v = self._project(x)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        out = math_attention(q, k, v, mask)
        return self._collect(out, B, T)


# ── Neighbor-sparse attention ─────────────────────────────────────────────────

SparseSelectorMode = Literal[
    "topology_only",
    "kmip_only",
    "symbolic_kmip",
    "symbolic_candidate_kmip",
]


class NeighborSparseMathAttention(_AttentionBase):
    """
    Multi-head attention that computes scores only for pre-selected neighbors.
    Complexity: O(T · K · D) instead of O(T² · D).

    forward(x, neighbors, valid)
      x         : (B, T, d_model)
      neighbors : (T, K) LongTensor — neighbor indices per token
      valid     : (T, K) BoolTensor  — True = real neighbor, False = padding
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        triton_block_d: int | None = None,
        triton_block_k: int | None = None,
    ) -> None:
        super().__init__(d_model, n_heads, dropout)
        self._qkv_weight_cache: torch.Tensor | None = None
        self._qkv_weight_versions: tuple[int, int, int] | None = None
        self.triton_block_d = triton_block_d
        self.triton_block_k = triton_block_k
        self.enable_fused_norm_qkv = False
        self.enable_fused_attn_outproj = False
        self.enable_block_token_attention = True
        self.block_token_attention_t = 2

    def _fused_qkv_weight(self) -> torch.Tensor:
        if torch.is_grad_enabled() and self.training:
            return torch.cat(
                (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight),
                dim=0,
            )
        versions = (
            self.q_proj.weight._version,
            self.k_proj.weight._version,
            self.v_proj.weight._version,
        )
        cache = self._qkv_weight_cache
        if (
            cache is None
            or self._qkv_weight_versions != versions
            or cache.device != self.q_proj.weight.device
            or cache.dtype != self.q_proj.weight.dtype
        ):
            self._qkv_weight_cache = torch.cat(
                (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight),
                dim=0,
            ).detach()
            self._qkv_weight_versions = versions
        return self._qkv_weight_cache

    def _project_qkv_bhtd(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_head
        qkv = F.linear(x, self._fused_qkv_weight())
        q, k, v = qkv.view(B, T, 3, H, Dh).unbind(dim=2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward_fused_norm_qkv(
        self,
        x: torch.Tensor,
        norm: nn.LayerNorm,
        neighbors: torch.Tensor,
        valid: torch.Tensor | None = None,
        selector_mode: SparseSelectorMode = "topology_only",
        symbolic_scores: torch.Tensor | None = None,
        selector_alpha: float = 1.0,
        selector_beta: float = 1.0,
        selector_k: int | None = None,
    ) -> torch.Tensor:
        if (
            not self.enable_fused_norm_qkv
            or self.training
            or selector_mode != "topology_only"
            or not x.is_cuda
        ):
            return self(
                norm(x), neighbors, valid,
                selector_mode=selector_mode,
                symbolic_scores=symbolic_scores,
                selector_alpha=selector_alpha,
                selector_beta=selector_beta,
                selector_k=selector_k,
            )
        from .triton_attention import (
            triton_layernorm_qkv,
            triton_neighbor_attention_flat,
            TRITON_AVAILABLE,
        )
        if not TRITON_AVAILABLE:
            return self(
                norm(x), neighbors, valid,
                selector_mode=selector_mode,
                symbolic_scores=symbolic_scores,
                selector_alpha=selector_alpha,
                selector_beta=selector_beta,
                selector_k=selector_k,
            )
        q, k, v = triton_layernorm_qkv(
            x, self._fused_qkv_weight(), norm.weight, norm.bias, norm.eps, self.n_heads
        )
        out = triton_neighbor_attention_flat(
            q, k, v, neighbors, valid,
            block_d=self.triton_block_d,
            block_k=self.triton_block_k,
        )
        return self.out_proj(self._maybe_dropout(out))

    def project_norm_qkv_for_profile(
        self, x: torch.Tensor, norm: nn.LayerNorm
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from .triton_attention import triton_layernorm_qkv
        return triton_layernorm_qkv(
            x, self._fused_qkv_weight(), norm.weight, norm.bias, norm.eps, self.n_heads
        )

    def forward(
        self,
        x: torch.Tensor,
        neighbors: torch.Tensor,
        valid: torch.Tensor | None = None,
        selector_mode: SparseSelectorMode = "topology_only",
        symbolic_scores: torch.Tensor | None = None,
        selector_alpha: float = 1.0,
        selector_beta: float = 1.0,
        selector_k: int | None = None,
    ) -> torch.Tensor:
        from .triton_attention import (
            triton_neighbor_attention_flat,
            TRITON_AVAILABLE,
        )
        from .sparse_attention import (
            neighbor_attention,
            neighbors_from_candidate_qk_scores,
            neighbors_from_qk_scores,
        )
        B, T, _ = x.shape
        q, k, v = self._project_qkv_bhtd(x)
        if selector_mode != "topology_only":
            final_k = selector_k if selector_k is not None else neighbors.shape[1]
            if selector_mode == "symbolic_candidate_kmip":
                neighbors, valid_bool = neighbors_from_candidate_qk_scores(
                    q, k,
                    candidate_neighbors=neighbors,
                    candidate_valid=valid.bool() if valid is not None else None,
                    max_k=final_k,
                    symbolic_scores=symbolic_scores,
                    alpha=selector_alpha,
                    beta=selector_beta,
                )
            else:
                neighbors, valid_bool = neighbors_from_qk_scores(
                    q, k,
                    max_k=final_k,
                    symbolic_scores=symbolic_scores if selector_mode == "symbolic_kmip" else None,
                    alpha=selector_alpha,
                    beta=selector_beta,
                )
            valid = valid_bool.char() if x.is_cuda else valid_bool
        if (not TRITON_AVAILABLE) or (not x.is_cuda):
            out = neighbor_attention(q, k, v, neighbors, valid.bool() if valid is not None else valid)
            out = out.transpose(1, 2).reshape(B, T, self.d_model)
        else:
            if self.enable_fused_attn_outproj and not self.training:
                from .triton_attention import triton_neighbor_attention_outproj
                return triton_neighbor_attention_outproj(
                    q, k, v, neighbors, valid,
                    self.out_proj.weight, self.out_proj.bias,
                    block_d=self.triton_block_d,
                    block_k=self.triton_block_k,
                )
            else:
                if self.enable_block_token_attention:
                    from .triton_attention import triton_neighbor_attention_flat_block_t
                    out = triton_neighbor_attention_flat_block_t(
                        q, k, v, neighbors, valid,
                        block_t=self.block_token_attention_t,
                        block_d=self.triton_block_d,
                        block_k=self.triton_block_k,
                    )
                else:
                    out = triton_neighbor_attention_flat(
                        q, k, v, neighbors, valid,
                        block_d=self.triton_block_d,
                        block_k=self.triton_block_k,
                    )
        return self.out_proj(self._maybe_dropout(out))

    def project_qkv_for_profile(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Expose fused QKV projection for block-level profiling."""
        return self._project_qkv_bhtd(x)

    def out_project_for_profile(self, out: torch.Tensor) -> torch.Tensor:
        """Expose output projection for block-level profiling."""
        return self.out_proj(self._maybe_dropout(out))

    def attention_out_project_for_profile(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        neighbors: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        from .triton_attention import triton_neighbor_attention_outproj
        return triton_neighbor_attention_outproj(
            q, k, v, neighbors, valid,
            self.out_proj.weight, self.out_proj.bias,
            block_d=self.triton_block_d,
            block_k=self.triton_block_k,
        )


class BlockSparseMathAttention(_AttentionBase):
    """Multi-head attention that consumes block-neighbor topology directly."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        use_fast: bool = True,
    ) -> None:
        super().__init__(d_model, n_heads, dropout)
        self.use_fast = bool(use_fast)
        self._qkv_weight_cache: torch.Tensor | None = None
        self._qkv_weight_versions: tuple[int, int, int] | None = None

    def _fused_qkv_weight(self) -> torch.Tensor:
        if torch.is_grad_enabled() and self.training:
            return torch.cat(
                (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight),
                dim=0,
            )
        versions = (
            self.q_proj.weight._version,
            self.k_proj.weight._version,
            self.v_proj.weight._version,
        )
        cache = self._qkv_weight_cache
        if (
            cache is None
            or self._qkv_weight_versions != versions
            or cache.device != self.q_proj.weight.device
            or cache.dtype != self.q_proj.weight.dtype
        ):
            self._qkv_weight_cache = torch.cat(
                (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight),
                dim=0,
            ).detach()
            self._qkv_weight_versions = versions
        return self._qkv_weight_cache

    def project_norm_qkv_for_profile(
        self, x: torch.Tensor, norm: nn.LayerNorm
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.is_cuda:
            from .triton_attention import TRITON_AVAILABLE, triton_layernorm_qkv
            if TRITON_AVAILABLE:
                return triton_layernorm_qkv(
                    x, self._fused_qkv_weight(), norm.weight, norm.bias, norm.eps, self.n_heads
                )
        return self._project(norm(x))

    def _block_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_neighbors: torch.Tensor,
        block_valid_i8: torch.Tensor | None,
        block_size: int,
        block_token_indices: torch.Tensor | None = None,
        block_token_valid_i8: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_fast:
            from .block_sparse_attention import block_sparse_attention_fast
            return block_sparse_attention_fast(
                q, k, v, block_neighbors, block_valid_i8, block_size,
                block_token_indices=block_token_indices,
                block_token_valid_i8=block_token_valid_i8,
            )
        from .block_sparse_attention import block_sparse_attention_reference
        return block_sparse_attention_reference(q, k, v, block_neighbors, block_valid_i8, block_size)

    def forward(
        self,
        x: torch.Tensor,
        block_neighbors: torch.Tensor,
        block_valid_i8: torch.Tensor | None,
        block_size: int,
        block_token_indices: torch.Tensor | None = None,
        block_token_valid_i8: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self._project(x)
        out = self._block_attention(
            q, k, v, block_neighbors, block_valid_i8, block_size,
            block_token_indices=block_token_indices,
            block_token_valid_i8=block_token_valid_i8,
        )
        return self._collect(out, B, T)

    def project_qkv_for_profile(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._project(x)

    def block_attention_for_profile(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_neighbors: torch.Tensor,
        block_valid_i8: torch.Tensor | None,
        block_size: int,
        block_token_indices: torch.Tensor | None = None,
        block_token_valid_i8: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._block_attention(
            q, k, v, block_neighbors, block_valid_i8, block_size,
            block_token_indices=block_token_indices,
            block_token_valid_i8=block_token_valid_i8,
        )

    def attention_out_project_for_profile(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_neighbors: torch.Tensor,
        block_valid_i8: torch.Tensor | None,
        block_size: int,
        block_token_indices: torch.Tensor | None = None,
        block_token_valid_i8: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out_bhtd = self._block_attention(
            q, k, v, block_neighbors, block_valid_i8, block_size,
            block_token_indices=block_token_indices,
            block_token_valid_i8=block_token_valid_i8,
        )
        return self._collect_bhtd(out_bhtd)

    def out_project_for_profile(self, out: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self._maybe_dropout(out))


# ── Aliases / backwards compat ────────────────────────────────────────────────

MathRoutedAttention = DenseMaskedMathAttention


class FullAttention(nn.Module):
    """Unconstrained baseline for benchmarking."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = DenseMaskedMathAttention(d_model, n_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x, mask=None)
