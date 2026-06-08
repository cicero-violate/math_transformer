from __future__ import annotations
import math
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

    def _collect(self, out: torch.Tensor, B: int, T: int) -> torch.Tensor:
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(self.dropout(out))

    def _project_bhtd(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H, Dh = self.n_heads, self.d_head
        def _p(proj: nn.Linear) -> torch.Tensor:
            w = proj.weight.view(H, Dh, self.d_model)
            return torch.einsum("btd,hfd->bhtf", x, w)
        return _p(self.q_proj), _p(self.k_proj), _p(self.v_proj)

    def _collect_bhtd(self, out: torch.Tensor) -> torch.Tensor:
        H, Dh = self.n_heads, self.d_head
        w = self.out_proj.weight.view(self.d_model, H, Dh)
        y = torch.einsum("bhtd,ohd->bto", self.dropout(out), w)
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

class NeighborSparseMathAttention(_AttentionBase):
    """
    Multi-head attention that computes scores only for pre-selected neighbors.
    Complexity: O(T · K · D) instead of O(T² · D).

    forward(x, neighbors, valid)
      x         : (B, T, d_model)
      neighbors : (T, K) LongTensor — neighbor indices per token
      valid     : (T, K) BoolTensor  — True = real neighbor, False = padding
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__(d_model, n_heads, dropout)
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

    def _project_qkv_bhtd(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        H, Dh = self.n_heads, self.d_head
        qkv = F.linear(x, self._fused_qkv_weight())
        q, k, v = qkv.view(B, T, 3, H, Dh).unbind(dim=2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        neighbors: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from .triton_attention import (
            triton_neighbor_attention_flat,
            TRITON_AVAILABLE,
        )
        from .sparse_attention import neighbor_attention
        B, T, _ = x.shape
        q, k, v = self._project_qkv_bhtd(x)
        if (not TRITON_AVAILABLE) or (not x.is_cuda):
            out = neighbor_attention(q, k, v, neighbors, valid.bool() if valid is not None else valid)
            out = out.transpose(1, 2).reshape(B, T, self.d_model)
        else:
            out = triton_neighbor_attention_flat(q, k, v, neighbors, valid)
        return self.out_proj(self.dropout(out))


# ── Aliases / backwards compat ────────────────────────────────────────────────

MathRoutedAttention = DenseMaskedMathAttention


class FullAttention(nn.Module):
    """Unconstrained baseline for benchmarking."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = DenseMaskedMathAttention(d_model, n_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x, mask=None)
