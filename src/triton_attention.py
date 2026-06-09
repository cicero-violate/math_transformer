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
        h   = pid % H
        bt  = pid // H
        t   = bt % T
        b   = bt // T

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


    @triton.jit
    def _nbr_sparse_attn_flat_kernel(
        Q_ptr, K_ptr, V_ptr,
        Nb_ptr, Vld_ptr,
        Out_ptr,
        q_sb, q_sh, q_st,
        k_sb, k_sh, k_st,
        v_sb, v_sh, v_st,
        o_sb, o_st,
        n_st,
        T, H, D, K,
        scale,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        t   = pid % T
        bh  = pid // T
        b   = bh // H
        h   = bh % H

        d_idx = tl.arange(0, BLOCK_D)
        k_idx = tl.arange(0, BLOCK_K)

        q = tl.load(
            Q_ptr + b * q_sb + h * q_sh + t * q_st + d_idx,
            mask=d_idx < D, other=0.0,
        )

        nb  = tl.load(Nb_ptr  + t * n_st + k_idx, mask=k_idx < K, other=0)
        vld = tl.load(Vld_ptr + t * n_st + k_idx, mask=k_idx < K, other=0).to(tl.float32)

        k_ptrs = K_ptr + b * k_sb + h * k_sh + nb[:, None] * k_st + d_idx[None, :]
        k_vecs = tl.load(
            k_ptrs,
            mask=(k_idx[:, None] < K) & (d_idx[None, :] < D),
            other=0.0,
        )

        scores = tl.sum(k_vecs * q[None, :], axis=1) * scale
        scores = tl.where((k_idx < K) & (vld > 0), scores, -float('inf'))

        s_max = tl.max(scores, axis=0)
        s_max = tl.where(s_max > -float('inf'), s_max, 0.0)
        scores_exp = tl.exp(scores - s_max)
        scores_exp = tl.where((k_idx < K) & (vld > 0), scores_exp, 0.0)
        s_sum = tl.sum(scores_exp, axis=0) + 1e-9
        probs = scores_exp / s_sum

        v_ptrs = V_ptr + b * v_sb + h * v_sh + nb[:, None] * v_st + d_idx[None, :]
        v_vecs = tl.load(
            v_ptrs,
            mask=(k_idx[:, None] < K) & (d_idx[None, :] < D),
            other=0.0,
        )

        out = tl.sum(probs[:, None] * v_vecs, axis=0)
        flat_d = h * D + d_idx
        tl.store(
            Out_ptr + b * o_sb + t * o_st + flat_d,
            out,
            mask=d_idx < D,
        )


    @triton.jit
    def _nbr_sparse_attn_flat_block_t_kernel(
        Q_ptr, K_ptr, V_ptr,
        Nb_ptr, Vld_ptr,
        Out_ptr,
        q_sb, q_sh, q_st,
        k_sb, k_sh, k_st,
        v_sb, v_sh, v_st,
        o_sb, o_st,
        n_st,
        T, H, D, K,
        scale,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        blocks_t = tl.cdiv(T, BLOCK_T)
        tb = pid % blocks_t
        bh = pid // blocks_t
        b = bh // H
        h = bh % H

        d_idx = tl.arange(0, BLOCK_D)
        k_idx = tl.arange(0, BLOCK_K)
        flat_d = h * D + d_idx

        for ti in tl.static_range(0, BLOCK_T):
            t = tb * BLOCK_T + ti
            t_mask = t < T
            q = tl.load(
                Q_ptr + b * q_sb + h * q_sh + t * q_st + d_idx,
                mask=t_mask & (d_idx < D),
                other=0.0,
            )
            nb = tl.load(Nb_ptr + t * n_st + k_idx, mask=t_mask & (k_idx < K), other=0)
            vld = tl.load(Vld_ptr + t * n_st + k_idx, mask=t_mask & (k_idx < K), other=0).to(tl.float32)
            k_vecs = tl.load(
                K_ptr + b * k_sb + h * k_sh + nb[:, None] * k_st + d_idx[None, :],
                mask=t_mask & (k_idx[:, None] < K) & (d_idx[None, :] < D),
                other=0.0,
            )
            scores = tl.sum(k_vecs * q[None, :], axis=1) * scale
            scores = tl.where(t_mask & (k_idx < K) & (vld > 0), scores, -float('inf'))
            s_max = tl.max(scores, axis=0)
            s_max = tl.where(s_max > -float('inf'), s_max, 0.0)
            scores_exp = tl.exp(scores - s_max)
            scores_exp = tl.where(t_mask & (k_idx < K) & (vld > 0), scores_exp, 0.0)
            probs = scores_exp / (tl.sum(scores_exp, axis=0) + 1e-9)
            v_vecs = tl.load(
                V_ptr + b * v_sb + h * v_sh + nb[:, None] * v_st + d_idx[None, :],
                mask=t_mask & (k_idx[:, None] < K) & (d_idx[None, :] < D),
                other=0.0,
            )
            out = tl.sum(probs[:, None] * v_vecs, axis=0)
            tl.store(
                Out_ptr + b * o_sb + t * o_st + flat_d,
                out,
                mask=t_mask & (d_idx < D),
            )


    @triton.jit
    def _layernorm_qkv_kernel(
        X_ptr, W_ptr, LnW_ptr, LnB_ptr,
        Q_ptr, K_ptr, V_ptr,
        x_sb, x_st, x_sd,
        q_sb, q_sh, q_st,
        k_sb, k_sh, k_st,
        v_sb, v_sh, v_st,
        T, D_MODEL, H, D_HEAD,
        eps,
        BLOCK_D: tl.constexpr,
        BLOCK_O: tl.constexpr,
    ):
        row = tl.program_id(0)
        ob = tl.program_id(1)
        t = row % T
        b = row // T
        d_idx = tl.arange(0, BLOCK_D)
        d_mask = d_idx < D_MODEL
        x = tl.load(X_ptr + b * x_sb + t * x_st + d_idx * x_sd, mask=d_mask, other=0.0).to(tl.float32)
        denom = D_MODEL + 0.0
        mean = tl.sum(tl.where(d_mask, x, 0.0), axis=0) / denom
        xc = tl.where(d_mask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / denom
        rstd = tl.rsqrt(var + eps)
        ln_w = tl.load(LnW_ptr + d_idx, mask=d_mask, other=0.0).to(tl.float32)
        ln_b = tl.load(LnB_ptr + d_idx, mask=d_mask, other=0.0).to(tl.float32)
        xn = tl.where(d_mask, (x - mean) * rstd * ln_w + ln_b, 0.0)
        o_idx = ob * BLOCK_O + tl.arange(0, BLOCK_O)
        o_mask = o_idx < (3 * D_MODEL)
        w = tl.load(W_ptr + o_idx[:, None] * D_MODEL + d_idx[None, :], mask=o_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        out = tl.sum(w * xn[None, :], axis=1)
        part = o_idx // D_MODEL
        rem = o_idx - part * D_MODEL
        h = rem // D_HEAD
        dh = rem - h * D_HEAD
        q_ptrs = Q_ptr + b * q_sb + h * q_sh + t * q_st + dh
        k_ptrs = K_ptr + b * k_sb + h * k_sh + t * k_st + dh
        v_ptrs = V_ptr + b * v_sb + h * v_sh + t * v_st + dh
        tl.store(q_ptrs, out, mask=o_mask & (part == 0) & (h < H) & (dh < D_HEAD))
        tl.store(k_ptrs, out, mask=o_mask & (part == 1) & (h < H) & (dh < D_HEAD))
        tl.store(v_ptrs, out, mask=o_mask & (part == 2) & (h < H) & (dh < D_HEAD))


    @triton.jit
    def _nbr_sparse_attn_outproj_kernel(
        Q_ptr, K_ptr, V_ptr,
        Nb_ptr, Vld_ptr,
        W_ptr, B_ptr,
        Out_ptr,
        q_sb, q_sh, q_st,
        k_sb, k_sh, k_st,
        v_sb, v_sh, v_st,
        w_so, w_si,
        o_sb, o_st,
        n_st,
        T, D_MODEL, D_HEAD, K_NBR,
        scale,
        BLOCK_H: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_O: tl.constexpr,
    ):
        row = tl.program_id(0)
        t = row % T
        b = row // T
        o_idx = tl.arange(0, BLOCK_O)
        d_idx = tl.arange(0, BLOCK_D)
        k_idx = tl.arange(0, BLOCK_K)
        o_mask = o_idx < D_MODEL

        acc = tl.load(B_ptr + o_idx, mask=o_mask, other=0.0).to(tl.float32)
        nb = tl.load(Nb_ptr + t * n_st + k_idx, mask=k_idx < K_NBR, other=0)
        vld = tl.load(Vld_ptr + t * n_st + k_idx, mask=k_idx < K_NBR, other=0).to(tl.float32)

        for h in tl.static_range(0, BLOCK_H):
            q = tl.load(
                Q_ptr + b * q_sb + h * q_sh + t * q_st + d_idx,
                mask=d_idx < D_HEAD,
                other=0.0,
            )
            k_vecs = tl.load(
                K_ptr + b * k_sb + h * k_sh + nb[:, None] * k_st + d_idx[None, :],
                mask=(k_idx[:, None] < K_NBR) & (d_idx[None, :] < D_HEAD),
                other=0.0,
            )
            scores = tl.sum(k_vecs * q[None, :], axis=1) * scale
            scores = tl.where((k_idx < K_NBR) & (vld > 0), scores, -float('inf'))
            s_max = tl.max(scores, axis=0)
            s_max = tl.where(s_max > -float('inf'), s_max, 0.0)
            scores_exp = tl.exp(scores - s_max)
            scores_exp = tl.where((k_idx < K_NBR) & (vld > 0), scores_exp, 0.0)
            probs = scores_exp / (tl.sum(scores_exp, axis=0) + 1e-9)
            v_vecs = tl.load(
                V_ptr + b * v_sb + h * v_sh + nb[:, None] * v_st + d_idx[None, :],
                mask=(k_idx[:, None] < K_NBR) & (d_idx[None, :] < D_HEAD),
                other=0.0,
            )
            attn = tl.sum(probs[:, None] * v_vecs, axis=0)
            in_idx = h * D_HEAD + d_idx
            w = tl.load(
                W_ptr + o_idx[:, None] * w_so + in_idx[None, :] * w_si,
                mask=o_mask[:, None] & (d_idx[None, :] < D_HEAD),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(w * attn[None, :], axis=1)

        tl.store(Out_ptr + b * o_sb + t * o_st + o_idx, acc, mask=o_mask)


def triton_layernorm_qkv(
    x: torch.Tensor,
    qkv_weight: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: float,
    n_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton is not installed")
    if not x.is_cuda or not qkv_weight.is_cuda or not ln_weight.is_cuda or not ln_bias.is_cuda:
        raise RuntimeError("triton_layernorm_qkv requires CUDA tensors")
    B, T, D_MODEL = x.shape
    if qkv_weight.shape != (3 * D_MODEL, D_MODEL):
        raise RuntimeError("qkv_weight must have shape (3*D,D)")
    D_HEAD = D_MODEL // n_heads
    q = torch.empty((B, n_heads, T, D_HEAD), device=x.device, dtype=x.dtype)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    BLOCK_D = triton.next_power_of_2(D_MODEL)
    BLOCK_O = min(64, triton.next_power_of_2(D_MODEL))
    grid = (B * T, triton.cdiv(3 * D_MODEL, BLOCK_O))
    _layernorm_qkv_kernel[grid](
        x, qkv_weight, ln_weight, ln_bias, q, k, v,
        x.stride(0), x.stride(1), x.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        T=T, D_MODEL=D_MODEL, H=n_heads, D_HEAD=D_HEAD, eps=eps,
        BLOCK_D=BLOCK_D, BLOCK_O=BLOCK_O,
    )
    return q, k, v


def triton_neighbor_attention(
    q: torch.Tensor,           # (B, H, T, D) — contiguous on GPU
    k: torch.Tensor,
    v: torch.Tensor,
    neighbors: torch.Tensor,   # (T, K) LongTensor on GPU
    valid: torch.Tensor,       # (T, K) BoolTensor on GPU
    block_d: int | None = None,
    block_k: int | None = None,
) -> torch.Tensor:
    """
    Fused neighbor-sparse attention via Triton kernel.
    Drops the Python token loop; gather + dot + softmax + weighted-sum in one kernel.

    Raises RuntimeError if Triton/CUDA is unavailable or tensors are not CUDA tensors.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton is not installed")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda or not neighbors.is_cuda:
        raise RuntimeError("triton_neighbor_attention requires CUDA tensors")
    if valid is None or not valid.is_cuda:
        raise RuntimeError("triton_neighbor_attention requires a CUDA valid mask")

    B, H, T, D = q.shape
    K = neighbors.shape[1]

    valid_i8 = valid

    out = torch.empty_like(q)

    BLOCK_D = block_d if block_d is not None else triton.next_power_of_2(D)
    BLOCK_K = block_k if block_k is not None else triton.next_power_of_2(K)
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


def triton_neighbor_attention_flat(
    q: torch.Tensor,           # (B, H, T, D), strided CUDA tensor accepted
    k: torch.Tensor,
    v: torch.Tensor,
    neighbors: torch.Tensor,   # (T, K) LongTensor on GPU
    valid: torch.Tensor,       # (T, K) int8 mask on GPU
    block_d: int | None = None,
    block_k: int | None = None,
) -> torch.Tensor:
    """Return sparse attention as (B, T, H*D) without an extra collect einsum."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton is not installed")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda or not neighbors.is_cuda:
        raise RuntimeError("triton_neighbor_attention_flat requires CUDA tensors")
    if valid is None or not valid.is_cuda:
        raise RuntimeError("triton_neighbor_attention_flat requires a CUDA valid mask")

    B, H, T, D = q.shape
    K = neighbors.shape[1]
    out = torch.empty((B, T, H * D), device=q.device, dtype=q.dtype)

    BLOCK_D = block_d if block_d is not None else triton.next_power_of_2(D)
    BLOCK_K = block_k if block_k is not None else triton.next_power_of_2(K)
    grid = (B * H * T,)

    _nbr_sparse_attn_flat_kernel[grid](
        q, k, v,
        neighbors, valid,
        out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1),
        neighbors.stride(0),
        T=T, H=H, D=D, K=K,
        scale=1.0 / math.sqrt(D),
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
    )
    return out


def triton_neighbor_attention_flat_block_t(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    neighbors: torch.Tensor,
    valid: torch.Tensor,
    block_t: int = 2,
    block_d: int | None = None,
    block_k: int | None = None,
) -> torch.Tensor:
    """Return sparse attention as (B,T,H*D), grouping multiple tokens per program."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton is not installed")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda or not neighbors.is_cuda:
        raise RuntimeError("triton_neighbor_attention_flat_block_t requires CUDA tensors")
    if valid is None or not valid.is_cuda:
        raise RuntimeError("triton_neighbor_attention_flat_block_t requires a CUDA valid mask")
    B, H, T, D = q.shape
    K = neighbors.shape[1]
    out = torch.empty((B, T, H * D), device=q.device, dtype=q.dtype)
    BLOCK_T = int(block_t)
    if BLOCK_T <= 0:
        raise RuntimeError("block_t must be positive")
    BLOCK_D = block_d if block_d is not None else triton.next_power_of_2(D)
    BLOCK_K = block_k if block_k is not None else triton.next_power_of_2(K)
    grid = (B * H * triton.cdiv(T, BLOCK_T),)
    _nbr_sparse_attn_flat_block_t_kernel[grid](
        q, k, v,
        neighbors, valid,
        out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1),
        neighbors.stride(0),
        T=T, H=H, D=D, K=K,
        scale=1.0 / math.sqrt(D),
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
    )
    return out


def triton_neighbor_attention_outproj(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    neighbors: torch.Tensor,
    valid: torch.Tensor,
    out_weight: torch.Tensor,
    out_bias: torch.Tensor,
    block_d: int | None = None,
    block_k: int | None = None,
    block_o: int | None = None,
) -> torch.Tensor:
    """Fused neighbor-sparse attention + output projection.

    Returns (B,T,D_MODEL), equivalent to
    `F.linear(triton_neighbor_attention_flat(q,k,v,...), out_weight, out_bias)`.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton is not installed")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda or not neighbors.is_cuda:
        raise RuntimeError("triton_neighbor_attention_outproj requires CUDA tensors")
    if valid is None or not valid.is_cuda:
        raise RuntimeError("triton_neighbor_attention_outproj requires a CUDA valid mask")
    if not out_weight.is_cuda or not out_bias.is_cuda:
        raise RuntimeError("triton_neighbor_attention_outproj requires CUDA output projection")

    B, H, T, D_HEAD = q.shape
    D_MODEL = H * D_HEAD
    if out_weight.shape != (D_MODEL, D_MODEL):
        raise RuntimeError("out_weight must have shape (D_MODEL,D_MODEL)")
    if out_bias.shape != (D_MODEL,):
        raise RuntimeError("out_bias must have shape (D_MODEL,)")

    K_NBR = neighbors.shape[1]
    out = torch.empty((B, T, D_MODEL), device=q.device, dtype=q.dtype)
    BLOCK_D = block_d if block_d is not None else triton.next_power_of_2(D_HEAD)
    BLOCK_K = block_k if block_k is not None else triton.next_power_of_2(K_NBR)
    BLOCK_O = block_o if block_o is not None else triton.next_power_of_2(D_MODEL)
    grid = (B * T,)
    _nbr_sparse_attn_outproj_kernel[grid](
        q, k, v,
        neighbors, valid,
        out_weight, out_bias,
        out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out_weight.stride(0), out_weight.stride(1),
        out.stride(0), out.stride(1),
        neighbors.stride(0),
        T=T, D_MODEL=D_MODEL, D_HEAD=D_HEAD, K_NBR=K_NBR,
        scale=1.0 / math.sqrt(D_HEAD),
        BLOCK_H=H,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
        BLOCK_O=BLOCK_O,
    )
    return out
