#!/usr/bin/env python
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import _collect_nodes_roots, _collect_nodes_trees, _load_env_from_examples
from src.model import MathRoutedTransformerBlock

DEFAULT_EXPRS = [
    "add(matmul(A, x), b)",
    "add(matmul(W, h), c)",
    "grad(f, x)",
    "sum(i, x_i)",
    "matmul(Q, K)",
    "constraint(leq(matmul(A, x), b))",
]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    p = argparse.ArgumentParser(description="Profile v7 neighbor-sparse block timing buckets")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--node-mode", choices=["roots", "trees"], default="trees")
    p.add_argument("--device", default="cuda")
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--local-window", type=int, default=1)
    p.add_argument("--topology-mode", default="middle_preserving_topk", choices=["union", "scored_topk", "middle_preserving_topk"])
    p.add_argument("--fixed-k", type=int, default=16)
    p.add_argument("--max-neighbors", type=int, default=16)
    p.add_argument("--middle-bridge-width", type=int, default=1)
    p.add_argument("--triton-block-d", type=int, default=None)
    p.add_argument("--triton-block-k", type=int, default=None)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--examples", default=None)
    p.add_argument("--no-static", action="store_true", help="Do not pre-load static topology buffers")
    args = p.parse_args()

    dev = torch.device(args.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    torch.manual_seed(args.seed)
    nodes = (
        _collect_nodes_trees(DEFAULT_EXPRS, args.n)
        if args.node_mode == "trees"
        else _collect_nodes_roots(DEFAULT_EXPRS, args.n)
    )
    env = _load_env_from_examples(args.examples)

    block = MathRoutedTransformerBlock(
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=0.0,
        attention_mode="neighbor_sparse",
        max_neighbors=args.max_neighbors,
        topk=args.topk,
        local_window=args.local_window,
        topology_mode=args.topology_mode,
        fixed_k=args.fixed_k,
        middle_bridge_width=args.middle_bridge_width,
        triton_block_d=args.triton_block_d,
        triton_block_k=args.triton_block_k,
    ).to(dev)
    block.eval()

    if not args.no_static:
        prepared = block.prepare_static_topology(nodes, env or None, device=dev)
        print(
            f"prepared length={prepared.length} k={prepared.k} "
            f"memory_bytes={prepared.memory_bytes} device={prepared.device}"
        )
        print(f"middle_bridge={prepared.diagnostics.by_relation.get('middle_bridge', 0)}")

    x = torch.randn(1, args.n, args.d_model, device=dev)
    buckets: dict[str, list[float]] = {}

    with torch.no_grad():
        for _ in range(args.warmup):
            block.profile_cached_sparse_block(x, nodes, env or None)
        _sync(dev)
        for _ in range(args.iters):
            _, timings = block.profile_cached_sparse_block(x, nodes, env or None)
            for k, v in timings.items():
                buckets.setdefault(k, []).append(float(v))

    print(
        f"profile n={args.n} node_mode={args.node_mode} device={dev} "
        f"topology_mode={args.topology_mode} fixed_k={args.fixed_k} max_neighbors={args.max_neighbors}"
    )
    print(f"triton_block_d={args.triton_block_d} triton_block_k={args.triton_block_k}")
    print("bucket                         mean_ms    median_ms")
    print("----------------------------------------------------")
    for key in [
        "topology_prepare_ms",
        "norm1_ms",
        "qkv_ms",
        "attention_kernel_ms",
        "out_proj_ms",
        "residual1_ms",
        "norm2_ms",
        "ffn_ms",
        "residual2_ms",
        "total_block_ms",
    ]:
        vals = buckets.get(key, [])
        if vals:
            print(f"{key:<30} {statistics.mean(vals):>8.3f} {statistics.median(vals):>12.3f}")


if __name__ == "__main__":
    main()
