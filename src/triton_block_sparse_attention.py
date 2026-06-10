from __future__ import annotations

import math

import torch

from .block_sparse_attention import block_sparse_attention_vectorized

try:  # pragma: no cover - availability only
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - CPU-only environments
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _block_sparse_attn_kernel(
        Q_ptr, K_ptr, V_ptr,
        BlockNb_ptr, BlockValid_ptr,
        Out_ptr,
        q_sb, q_sh, q_st,
        k_sb, k_sh, k_st,
        v_sb, v_sh, v_st,
        o_sb, o_sh, o_st,
        bn_s0,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        K_BLOCKS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_KT: tl.constexpr,
        scale: tl.constexpr,
    ):
        pid = tl.program_id(0)
        qb = pid % NUM_BLOCKS
        bh = pid // NUM_BLOCKS
        b = bh // H
        h = bh - b * H

        q_tok = qb * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        d_idx = tl.arange(0, BLOCK_D)
        kt_idx = tl.arange(0, BLOCK_KT)
        kb_slot = kt_idx // BLOCK_SIZE
        kt_in_block = kt_idx - kb_slot * BLOCK_SIZE

        q = tl.load(
            Q_ptr + b * q_sb + h * q_sh + q_tok[:, None] * q_st + d_idx[None, :],
            mask=(q_tok[:, None] < T) & (d_idx[None, :] < D),
            other=0.0,
        )

        nb = tl.load(
            BlockNb_ptr + qb * bn_s0 + kb_slot,
            mask=kb_slot < K_BLOCKS,
            other=0,
        )
        bv = tl.load(
            BlockValid_ptr + qb * bn_s0 + kb_slot,
            mask=kb_slot < K_BLOCKS,
            other=0,
        ).to(tl.int1)
        key_tok = nb * BLOCK_SIZE + kt_in_block
        key_ok = (kb_slot < K_BLOCKS) & bv & (nb >= 0) & (nb < NUM_BLOCKS) & (key_tok < T)

        k_tile = tl.load(
            K_ptr + b * k_sb + h * k_sh + key_tok[:, None] * k_st + d_idx[None, :],
            mask=key_ok[:, None] & (d_idx[None, :] < D),
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k_tile), input_precision="tf32") * scale
        scores = tl.where((q_tok[:, None] < T) & key_ok[None, :], scores, -float("inf"))

        m = tl.max(scores, axis=1)
        m = tl.where(m > -float("inf"), m, 0.0)
        p = tl.exp(scores - m[:, None])
        p = tl.where((q_tok[:, None] < T) & key_ok[None, :], p, 0.0)
        denom = tl.sum(p, axis=1) + 1.0e-9
        p = p / denom[:, None]

        v_tile = tl.load(
            V_ptr + b * v_sb + h * v_sh + key_tok[:, None] * v_st + d_idx[None, :],
            mask=key_ok[:, None] & (d_idx[None, :] < D),
            other=0.0,
        )
        out = tl.dot(p, v_tile, input_precision="tf32")
        tl.store(
            Out_ptr + b * o_sb + h * o_sh + q_tok[:, None] * o_st + d_idx[None, :],
            out,
            mask=(q_tok[:, None] < T) & (d_idx[None, :] < D),
        )

    @triton.jit
    def _block_sparse_attn_row_kernel(
        Q_ptr, K_ptr, V_ptr,
        BlockNb_ptr, BlockValid_ptr,
        Out_ptr,
        q_sb, q_sh, q_st,
        k_sb, k_sh, k_st,
        v_sb, v_sh, v_st,
        o_sb, o_sh, o_st,
        bn_s0,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        K_BLOCKS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_KT: tl.constexpr,
        scale: tl.constexpr,
    ):
        """One program per (batch, head, query token).

        This uses much less shared memory than the query-block program because
        scores/probabilities are rank-1 over K_BLOCKS * BLOCK_SIZE keys instead
        of rank-2 over BLOCK_SIZE query rows and K_BLOCKS * BLOCK_SIZE keys.
        """
        pid = tl.program_id(0)
        t = pid % T
        bh = pid // T
        b = bh // H
        h = bh - b * H
        qb = t // BLOCK_SIZE

        d_idx = tl.arange(0, BLOCK_D)
        kt_idx = tl.arange(0, BLOCK_KT)
        kb_slot = kt_idx // BLOCK_SIZE
        kt_in_block = kt_idx - kb_slot * BLOCK_SIZE

        q = tl.load(
            Q_ptr + b * q_sb + h * q_sh + t * q_st + d_idx,
            mask=d_idx < D,
            other=0.0,
        )
        nb = tl.load(
            BlockNb_ptr + qb * bn_s0 + kb_slot,
            mask=kb_slot < K_BLOCKS,
            other=0,
        )
        bv = tl.load(
            BlockValid_ptr + qb * bn_s0 + kb_slot,
            mask=kb_slot < K_BLOCKS,
            other=0,
        ).to(tl.int1)
        key_tok = nb * BLOCK_SIZE + kt_in_block
        key_ok = (kb_slot < K_BLOCKS) & bv & (nb >= 0) & (nb < NUM_BLOCKS) & (key_tok < T)

        k_vecs = tl.load(
            K_ptr + b * k_sb + h * k_sh + key_tok[:, None] * k_st + d_idx[None, :],
            mask=key_ok[:, None] & (d_idx[None, :] < D),
            other=0.0,
        )
        scores = tl.sum(k_vecs * q[None, :], axis=1) * scale
        scores = tl.where(key_ok, scores, -float("inf"))
        m = tl.max(scores, axis=0)
        m = tl.where(m > -float("inf"), m, 0.0)
        p = tl.exp(scores - m)
        p = tl.where(key_ok, p, 0.0)
        p = p / (tl.sum(p, axis=0) + 1.0e-9)

        v_vecs = tl.load(
            V_ptr + b * v_sb + h * v_sh + key_tok[:, None] * v_st + d_idx[None, :],
            mask=key_ok[:, None] & (d_idx[None, :] < D),
            other=0.0,
        )
        out = tl.sum(p[:, None] * v_vecs, axis=0)
        tl.store(
            Out_ptr + b * o_sb + h * o_sh + t * o_st + d_idx,
            out,
            mask=d_idx < D,
        )


    @triton.jit
    def _block_sparse_attn_token_row_kernel(
        Q_ptr, K_ptr, V_ptr,
        TokenIdx_ptr, TokenValid_ptr,
        Out_ptr,
        q_sb, q_sh, q_st,
        k_sb, k_sh, k_st,
        v_sb, v_sh, v_st,
        o_sb, o_sh, o_st,
        tok_s0, tok_s1,
        T: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        K_BLOCKS: tl.constexpr,
        TOKEN_CAP: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_KT: tl.constexpr,
        scale: tl.constexpr,
    ):
        pid = tl.program_id(0)
        t = pid % T
        bh = pid // T
        b = bh // H
        h = bh - b * H
        qb = t // BLOCK_SIZE

        d_idx = tl.arange(0, BLOCK_D)
        kt_idx = tl.arange(0, BLOCK_KT)
        kb_slot = kt_idx // TOKEN_CAP
        tc_slot = kt_idx - kb_slot * TOKEN_CAP

        q = tl.load(
            Q_ptr + b * q_sb + h * q_sh + t * q_st + d_idx,
            mask=d_idx < D,
            other=0.0,
        )
        key_tok = tl.load(
            TokenIdx_ptr + qb * tok_s0 + kb_slot * tok_s1 + tc_slot,
            mask=(kb_slot < K_BLOCKS) & (tc_slot < TOKEN_CAP),
            other=0,
        )
        tv = tl.load(
            TokenValid_ptr + qb * tok_s0 + kb_slot * tok_s1 + tc_slot,
            mask=(kb_slot < K_BLOCKS) & (tc_slot < TOKEN_CAP),
            other=0,
        ).to(tl.int1)
        key_ok = (kb_slot < K_BLOCKS) & (tc_slot < TOKEN_CAP) & tv & (key_tok >= 0) & (key_tok < T)
        k_vecs = tl.load(
            K_ptr + b * k_sb + h * k_sh + key_tok[:, None] * k_st + d_idx[None, :],
            mask=key_ok[:, None] & (d_idx[None, :] < D),
            other=0.0,
        )
        scores = tl.sum(k_vecs * q[None, :], axis=1) * scale
        scores = tl.where(key_ok, scores, -float("inf"))
        m = tl.max(scores, axis=0)
        m = tl.where(m > -float("inf"), m, 0.0)
        p = tl.exp(scores - m)
        p = tl.where(key_ok, p, 0.0)
        p = p / (tl.sum(p, axis=0) + 1.0e-9)
        v_vecs = tl.load(
            V_ptr + b * v_sb + h * v_sh + key_tok[:, None] * v_st + d_idx[None, :],
            mask=key_ok[:, None] & (d_idx[None, :] < D),
            other=0.0,
        )
        out = tl.sum(p[:, None] * v_vecs, axis=0)
        tl.store(
            Out_ptr + b * o_sb + h * o_sh + t * o_st + d_idx,
            out,
            mask=d_idx < D,
        )



def triton_block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_neighbors: torch.Tensor,
    block_valid_i8: torch.Tensor | None,
    block_size: int,
    *,
    block_token_indices: torch.Tensor | None = None,
    block_token_valid_i8: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused native block-sparse attention via Triton.

    Grid unit: one program per ``(batch, head, query_block)``. Each program loads
    a contiguous query block and the selected key/value blocks, performs stable
    softmax, and writes one output block. No token-neighbor table or gathered
    PyTorch SDPA tensor is materialized.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton is not installed")
    if not (q.is_cuda and k.is_cuda and v.is_cuda and block_neighbors.is_cuda):
        raise RuntimeError("triton_block_sparse_attention requires CUDA tensors")
    if block_valid_i8 is None:
        block_valid_i8 = torch.ones_like(block_neighbors, dtype=torch.int8, device=block_neighbors.device)
    if not block_valid_i8.is_cuda:
        raise RuntimeError("triton_block_sparse_attention requires CUDA block_valid_i8")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise RuntimeError("q, k, and v must have shape (batch, heads, N, d_head)")
    if q.shape != k.shape or q.shape != v.shape:
        raise RuntimeError(f"q, k, v shape mismatch: {q.shape}, {k.shape}, {v.shape}")
    if block_neighbors.ndim != 2 or block_valid_i8.shape != block_neighbors.shape:
        raise RuntimeError("block neighbor tensors must have matching rank-2 shapes")

    B, H, T, D = q.shape
    bs = int(block_size)
    if bs <= 0:
        raise RuntimeError("block_size must be positive")
    num_blocks = max(1, (T + bs - 1) // bs)
    if int(block_neighbors.shape[0]) < num_blocks:
        # Rare malformed/partial topology; preserve reference semantics.
        return block_sparse_attention_vectorized(q, k, v, block_neighbors, block_valid_i8, block_size)
    K_BLOCKS = int(block_neighbors.shape[1])
    if K_BLOCKS <= 0:
        return torch.zeros_like(q)

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    nb = block_neighbors[:num_blocks].to(device=q.device, dtype=torch.long).contiguous()
    bv = block_valid_i8[:num_blocks].to(device=q.device, dtype=torch.int8).contiguous()
    out = torch.empty_like(q_c)

    BLOCK_D = triton.next_power_of_2(D)
    grid = (B * H * T,)
    if block_token_indices is not None:
        tok = block_token_indices[:num_blocks].to(device=q.device, dtype=torch.long).contiguous()
        if tok.ndim != 3 or int(tok.shape[0]) < num_blocks or int(tok.shape[1]) != K_BLOCKS:
            raise RuntimeError("block_token_indices must have shape (num_blocks, K_BLOCKS, TOKEN_CAP)")
        if block_token_valid_i8 is None:
            tok_valid = torch.ones_like(tok, dtype=torch.int8, device=q.device)
        else:
            tok_valid = block_token_valid_i8[:num_blocks].to(device=q.device, dtype=torch.int8).contiguous()
        TOKEN_CAP = int(tok.shape[2])
        BLOCK_KT_TOKEN = triton.next_power_of_2(K_BLOCKS * TOKEN_CAP)
        _block_sparse_attn_token_row_kernel[grid](
            q_c, k_c, v_c,
            tok, tok_valid,
            out,
            q_c.stride(0), q_c.stride(1), q_c.stride(2),
            k_c.stride(0), k_c.stride(1), k_c.stride(2),
            v_c.stride(0), v_c.stride(1), v_c.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            tok.stride(0), tok.stride(1),
            T=T, H=H, D=D,
            K_BLOCKS=K_BLOCKS,
            TOKEN_CAP=TOKEN_CAP,
            BLOCK_SIZE=bs,
            BLOCK_D=BLOCK_D,
            BLOCK_KT=BLOCK_KT_TOKEN,
            scale=1.0 / math.sqrt(D),
            num_warps=4,
        )
        return out

    BLOCK_KT = triton.next_power_of_2(K_BLOCKS * bs)
    _block_sparse_attn_row_kernel[grid](
        q_c, k_c, v_c,
        nb, bv,
        out,
        q_c.stride(0), q_c.stride(1), q_c.stride(2),
        k_c.stride(0), k_c.stride(1), k_c.stride(2),
        v_c.stride(0), v_c.stride(1), v_c.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        nb.stride(0),
        T=T, H=H, D=D,
        NUM_BLOCKS=num_blocks,
        K_BLOCKS=K_BLOCKS,
        BLOCK_SIZE=bs,
        BLOCK_D=BLOCK_D,
        BLOCK_KT=BLOCK_KT,
        scale=1.0 / math.sqrt(D),
        num_warps=8,
    )
    return out
