"""
Sprint 5: Fused neighbor-sparse attention kernel in Triton.

Kernel: _nbr_sparse_attn_kernel
  - One program per (batch, head, token)
  - Gathers K neighbor keys/values into SRAM
  - Computes dot-product scores, stable softmax, weighted sum — all fused
  - Complexity: O(T * K * D) FLOPs, eliminates Python loop + K separate kernel launches

Gate: triton_sparse_attn_ms < dense_full_attn_ms at some n.
"""
from __future__ import annotations
import math

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _nbr_sparse_attn_kernel(
        Q_ptr, K_ptr, V_ptr,
        Nb_ptr, Vld_ptr,
        Out_ptr,
        # strides for (B, H, T, D) layout
        q_sb, q_sh, q_st,
        k_sb, k_sh, k_st,
        v_sb, v_sh, v_st,
        o_sb, o_sh, o_st,
        # stride for (T, K) neighbor / valid arrays
        n_st,
        # runtime dims
        T, H, D, K,
        scale,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """
        Grid: (B * H * T,)
        Each program handles one (batch, head, token) output row.
        """
        pid = tl.program_id(0)
        t   = pid % T
        bh  = pid // T
        b   = bh // H
        h   = bh % H

        d_idx = tl.arange(0, BLOCK_D)   # (BLOCK_D,)
        k_idx = tl.arange(0, BLOCK_K)   # (BLOCK_K,)

        # Load Q[b, h, t, :D]
        q = tl.load(
            Q_ptr + b * q_sb + h * q_sh + t * q_st + d_idx,
            mask=d_idx < D, other=0.0,
        )  # (BLOCK_D,)

        # Load neighbor indices (int64) and valid flags (int8→float)
        nb  = tl.load(Nb_ptr  + t * n_st + k_idx, mask=k_idx < K, other=0)
        vld = tl.load(Vld_ptr + t * n_st + k_idx, mask=k_idx < K, other=0).to(tl.float32)

        # Gather K key vectors: (BLOCK_K, BLOCK_D)
        k_ptrs = K_ptr + b * k_sb + h * k_sh + nb[:, None] * k_st + d_idx[None, :]
        k_vecs = tl.load(
            k_ptrs,
            mask=(k_idx[:, None] < K) & (d_idx[None, :] < D),
            other=0.0,
        )

        # Scores: row-wise dot(q, k_i) * scale → (BLOCK_K,)
        scores = tl.sum(k_vecs * q[None, :], axis=1) * scale
        scores = tl.where((k_idx < K) & (vld > 0), scores, -float('inf'))

        # Stable softmax
        s_max = tl.max(scores, axis=0)
        s_max = tl.where(s_max > -float('inf'), s_max, 0.0)  # guard all-masked row
        scores_exp = tl.exp(scores - s_max)
        scores_exp = tl.where((k_idx < K) & (vld > 0), scores_exp, 0.0)
        s_sum = tl.sum(scores_exp, axis=0) + 1e-9
        probs = scores_exp / s_sum  # (BLOCK_K,)

        # Gather V vectors: (BLOCK_K, BLOCK_D)
        v_ptrs = V_ptr + b * v_sb + h * v_sh + nb[:, None] * v_st + d_idx[None, :]
        v_vecs = tl.load(
            v_ptrs,
            mask=(k_idx[:, None] < K) & (d_idx[None, :] < D),
            other=0.0,
        )

        # Weighted sum: probs (BLOCK_K,) · v_vecs (BLOCK_K, BLOCK_D) → (BLOCK_D,)
        out = tl.sum(probs[:, None] * v_vecs, axis=0)

        # Store output
        tl.store(
            Out_ptr + b * o_sb + h * o_sh + t * o_st + d_idx,
            out,
            mask=d_idx < D,
        )


def triton_neighbor_attention(
    q: torch.Tensor,           # (B, H, T, D) — contiguous on GPU
    k: torch.Tensor,
    v: torch.Tensor,
    neighbors: torch.Tensor,   # (T, K) LongTensor on GPU
    valid: torch.Tensor,       # (T, K) BoolTensor on GPU
) -> torch.Tensor:
    """
    Fused neighbor-sparse attention via Triton kernel.
    Drops the Python token loop; gather + dot + softmax + weighted-sum in one kernel.

    Raises RuntimeError if Triton is not installed.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton is not installed")

    B, H, T, D = q.shape
    K = neighbors.shape[1]

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    neighbors = neighbors.contiguous()
    valid_i8 = valid.to(torch.int8).contiguous()

    out = torch.empty_like(q)

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_K = triton.next_power_of_2(K)
    grid = (B * H * T,)

    _nbr_sparse_attn_kernel[grid](
        q, k, v,
        neighbors, valid_i8,
        out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        neighbors.stride(0),
        T=T, H=H, D=D, K=K,
        scale=1.0 / math.sqrt(D),
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
    )
    return out
