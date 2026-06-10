from __future__ import annotations

import math

import torch
import torch.nn.functional as F

BLOCK_SPARSE_LAST_BACKEND = "none"


def block_sparse_attention_last_backend() -> str:
    return BLOCK_SPARSE_LAST_BACKEND


def _num_blocks(n_tokens: int, block_size: int) -> int:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return max(1, (int(n_tokens) + int(block_size) - 1) // int(block_size))


def block_neighbors_to_dense_mask(
    block_neighbors: torch.Tensor,
    block_valid_i8: torch.Tensor | None,
    block_size: int,
    n_tokens: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Materialize the dense token mask implied by block-neighbor topology.

    This is a correctness helper. It intentionally uses O(N^2) memory and should
    not be used on the hot path.
    """
    dev = device if device is not None else block_neighbors.device
    num_query_blocks = _num_blocks(n_tokens, block_size)
    mask = torch.zeros((n_tokens, n_tokens), dtype=torch.bool, device=dev)
    nb = block_neighbors.to(device=dev, dtype=torch.long)
    valid = (
        torch.ones_like(nb, dtype=torch.bool, device=dev)
        if block_valid_i8 is None
        else block_valid_i8.to(device=dev).bool()
    )
    row_count = min(num_query_blocks, int(nb.shape[0]))
    for bi in range(row_count):
        q_start = bi * block_size
        q_end = min(q_start + block_size, n_tokens)
        row = nb[bi]
        good = valid[bi] & (row >= 0) & (row < num_query_blocks)
        if not bool(good.any()):
            continue
        for bj in torch.unique(row[good], sorted=True).tolist():
            k_start = int(bj) * block_size
            k_end = min(k_start + block_size, n_tokens)
            if k_end > k_start:
                mask[q_start:q_end, k_start:k_end] = True
    return mask


def block_sparse_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_neighbors: torch.Tensor,
    block_valid_i8: torch.Tensor | None,
    block_size: int,
) -> torch.Tensor:
    """Reference native block-sparse attention.

    Parameters
    ----------
    q, k, v:
        Tensors shaped ``(batch, heads, N, d_head)``.
    block_neighbors:
        Long tensor shaped ``(B, K_B)`` containing selected key-block indices for
        each query block.
    block_valid_i8:
        Tensor shaped like ``block_neighbors``; nonzero entries are valid.
    block_size:
        Token count per block. The final block may be shorter.

    Returns
    -------
    torch.Tensor
        Output shaped ``(batch, heads, N, d_head)``.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape (batch, heads, N, d_head)")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, v shape mismatch: {q.shape}, {k.shape}, {v.shape}")
    if block_neighbors.ndim != 2:
        raise ValueError("block_neighbors must be rank-2")
    if block_valid_i8 is not None and block_valid_i8.shape != block_neighbors.shape:
        raise ValueError("block_valid_i8 must match block_neighbors shape")

    batch, heads, n_tokens, d_head = q.shape
    num_query_blocks = _num_blocks(n_tokens, block_size)
    nb = block_neighbors.to(device=q.device, dtype=torch.long)
    valid = (
        torch.ones_like(nb, dtype=torch.bool, device=q.device)
        if block_valid_i8 is None
        else block_valid_i8.to(device=q.device).bool()
    )
    out = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(d_head)
    row_count = min(num_query_blocks, int(nb.shape[0]))

    for bi in range(row_count):
        q_start = bi * block_size
        q_end = min(q_start + block_size, n_tokens)
        row = nb[bi]
        good = valid[bi] & (row >= 0) & (row < num_query_blocks)
        if not bool(good.any()):
            continue
        selected_blocks = torch.unique(row[good], sorted=True)
        idx_parts = []
        for bj in selected_blocks.tolist():
            k_start = int(bj) * block_size
            k_end = min(k_start + block_size, n_tokens)
            if k_end > k_start:
                idx_parts.append(torch.arange(k_start, k_end, device=q.device))
        if not idx_parts:
            continue
        key_idx = torch.cat(idx_parts, dim=0)
        q_blk = q[:, :, q_start:q_end, :]
        k_blk = k.index_select(2, key_idx)
        v_blk = v.index_select(2, key_idx)
        scores = torch.matmul(q_blk, k_blk.transpose(-2, -1)) * scale
        probs = F.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        out[:, :, q_start:q_end, :] = torch.matmul(probs, v_blk)

    return out


def block_sparse_attention_fast(
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
    """Fast native block-sparse entry point.

    CUDA dispatches to the Triton block kernel. CPU, or CUDA with Triton absent,
    uses the vectorized PyTorch fallback. Triton kernel failures are intentionally
    not swallowed; benchmark runs should fail loudly instead of silently reporting
    fallback timings as native-kernel timings.
    """
    global BLOCK_SPARSE_LAST_BACKEND
    if q.is_cuda:
        from .triton_block_sparse_attention import TRITON_AVAILABLE, triton_block_sparse_attention
        if TRITON_AVAILABLE:
            BLOCK_SPARSE_LAST_BACKEND = "triton"
            return triton_block_sparse_attention(
                q, k, v, block_neighbors, block_valid_i8, block_size,
                block_token_indices=block_token_indices,
                block_token_valid_i8=block_token_valid_i8,
            )
    BLOCK_SPARSE_LAST_BACKEND = "vectorized"
    return block_sparse_attention_vectorized(
        q, k, v, block_neighbors, block_valid_i8, block_size,
        block_token_indices=block_token_indices,
        block_token_valid_i8=block_token_valid_i8,
    )


def block_sparse_attention_vectorized(
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
    """Vectorized native block-sparse attention fallback.

    This path keeps the block topology native: it gathers contiguous key/value
    block tiles directly from ``block_neighbors`` and evaluates all query blocks
    in one batched attention call. It intentionally does **not** expand the
    topology into an ``(N, K)`` token-neighbor table.

    For the v15 benchmark shape ``N=4096, block_size=64, B=64`` this replaces
    the reference implementation's 64 Python-loop matmul/softmax calls with one
    batched block operation.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape (batch, heads, N, d_head)")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, v shape mismatch: {q.shape}, {k.shape}, {v.shape}")
    if block_neighbors.ndim != 2:
        raise ValueError("block_neighbors must be rank-2")
    if block_valid_i8 is not None and block_valid_i8.shape != block_neighbors.shape:
        raise ValueError("block_valid_i8 must match block_neighbors shape")

    batch, heads, n_tokens, d_head = q.shape
    num_blocks = _num_blocks(n_tokens, block_size)
    padded_n = num_blocks * int(block_size)
    pad_n = padded_n - n_tokens

    if pad_n:
        q_pad = F.pad(q, (0, 0, 0, pad_n))
        k_pad = F.pad(k, (0, 0, 0, pad_n))
        v_pad = F.pad(v, (0, 0, 0, pad_n))
    else:
        q_pad = q.contiguous()
        k_pad = k.contiguous()
        v_pad = v.contiguous()

    nb_src = block_neighbors.to(device=q.device, dtype=torch.long)
    if block_valid_i8 is None:
        valid_src = torch.ones_like(nb_src, dtype=torch.bool, device=q.device)
    else:
        valid_src = block_valid_i8.to(device=q.device).bool()

    # Match the reference behavior for missing query-block rows: they produce
    # zero output. Normal prepared block topology has exactly ``num_blocks`` rows.
    if nb_src.shape[0] < num_blocks:
        row_pad = num_blocks - int(nb_src.shape[0])
        nb_src = torch.cat([nb_src, torch.zeros(row_pad, nb_src.shape[1], dtype=torch.long, device=q.device)], dim=0)
        valid_src = torch.cat([valid_src, torch.zeros(row_pad, valid_src.shape[1], dtype=torch.bool, device=q.device)], dim=0)
    elif nb_src.shape[0] > num_blocks:
        nb_src = nb_src[:num_blocks]
        valid_src = valid_src[:num_blocks]

    key_blocks = nb_src.shape[1]
    if block_token_indices is not None:
        token_idx = block_token_indices.to(device=q.device, dtype=torch.long)
        if token_idx.ndim != 3 or token_idx.shape[0] < num_blocks or token_idx.shape[1] != key_blocks:
            raise ValueError("block_token_indices must have shape (num_blocks, key_blocks, cap)")
        token_idx = token_idx[:num_blocks]
        if block_token_valid_i8 is None:
            token_valid = torch.ones_like(token_idx, dtype=torch.bool, device=q.device)
        else:
            token_valid = block_token_valid_i8.to(device=q.device).bool()[:num_blocks]
        block_ok = valid_src & (nb_src >= 0) & (nb_src < num_blocks)
        token_ok = token_valid & block_ok.unsqueeze(-1) & (token_idx >= 0) & (token_idx < n_tokens)
        safe_token_idx = token_idx.clamp(0, max(n_tokens - 1, 0))
        flat_token_idx = safe_token_idx.reshape(-1)
        key_tokens_per_query_block = token_idx.shape[1] * token_idx.shape[2]
    else:
        safe_nb = nb_src.clamp(0, max(num_blocks - 1, 0))
        block_ok = valid_src & (nb_src >= 0) & (nb_src < num_blocks)
        offsets = torch.arange(int(block_size), device=q.device, dtype=torch.long)
        token_idx = safe_nb.unsqueeze(-1) * int(block_size) + offsets
        token_ok = block_ok.unsqueeze(-1) & (token_idx < n_tokens)
        flat_token_idx = token_idx.reshape(-1)
        key_tokens_per_query_block = key_blocks * int(block_size)

    q_blocks = q_pad.view(batch, heads, num_blocks, int(block_size), d_head)
    k_sel = k_pad.index_select(2, flat_token_idx).view(
        batch, heads, num_blocks, key_tokens_per_query_block, d_head
    )
    v_sel = v_pad.index_select(2, flat_token_idx).view(
        batch, heads, num_blocks, key_tokens_per_query_block, d_head
    )

    q_mat = q_blocks.reshape(batch * heads * num_blocks, int(block_size), d_head)
    k_mat = k_sel.reshape(batch * heads * num_blocks, key_tokens_per_query_block, d_head)
    v_mat = v_sel.reshape(batch * heads * num_blocks, key_tokens_per_query_block, d_head)
    attn_mask = token_ok.reshape(num_blocks, key_tokens_per_query_block)
    attn_mask = attn_mask.view(1, 1, num_blocks, 1, key_tokens_per_query_block)
    attn_mask = attn_mask.expand(batch, heads, num_blocks, 1, key_tokens_per_query_block)
    attn_mask = attn_mask.reshape(batch * heads * num_blocks, 1, key_tokens_per_query_block)

    # CUDA uses PyTorch's fused scaled-dot-product attention backend when
    # available. CPU keeps an explicit bmm/softmax path for stable test behavior.
    if q.is_cuda and hasattr(F, "scaled_dot_product_attention"):
        out_mat = F.scaled_dot_product_attention(
            q_mat, k_mat, v_mat,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
    else:
        scores = torch.bmm(q_mat, k_mat.transpose(1, 2)) * (1.0 / math.sqrt(d_head))
        scores = scores.masked_fill(~attn_mask, float("-inf"))
        probs = F.softmax(scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        out_mat = torch.bmm(probs, v_mat)

    out = out_mat.view(batch, heads, num_blocks, int(block_size), d_head)
    out = out.reshape(batch, heads, padded_n, d_head)[:, :, :n_tokens, :]
    return out.contiguous()
