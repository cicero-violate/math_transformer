from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch


# ── Timing helpers ────────────────────────────────────────────────────────────

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_runs(fn, n_warmup: int, n_iter: int) -> dict[str, float]:
    """
    Return timing stats (ms) for fn() over n_iter calls after n_warmup warmup runs.
    Returns: min_ms, median_ms, p95_ms, mean_ms, std_ms.
    Calls cuda.synchronize() around each timed call when on GPU.
    """
    with torch.no_grad():
        for _ in range(n_warmup):
            fn()
            _sync()
    times = []
    with torch.no_grad():
        for _ in range(n_iter):
            _sync()
            t0 = time.perf_counter()
            fn()
            _sync()
            times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return {
        "min_ms":    float(arr.min()),
        "median_ms": float(np.median(arr)),
        "p95_ms":    float(np.percentile(arr, 95)),
        "mean_ms":   float(arr.mean()),
        "std_ms":    float(arr.std()),
    }


def _timed(fn, n_warmup: int = 3, n_iter: int = 10) -> float:
    """Return median ms."""
    return _time_runs(fn, n_warmup, n_iter)["median_ms"]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class BenchmarkReport:
    n: int
    node_mode: str
    k: int
    full_edges: int
    allowed_edges: int
    avg_k: float
    max_k: int
    padding_ratio: float
    sparsity_ratio: float
    relation_reduction: float
    device: str = "cpu"
    # Sprint 1: topology build time (CPU path, cache-miss cost)
    topology_build_ms: float = 0.0
    # Attention-only
    dense_full_attn_ms: float = 0.0
    dense_masked_attn_ms: float = 0.0
    nbr_sparse_exact_ms: float = 0.0
    nbr_sparse_trunc_ms: float = 0.0
    # Sprint 3: torch.compile sparse kernel
    compiled_sparse_attn_ms: float = 0.0
    # Sprint 5: Triton fused kernel
    triton_sparse_attn_ms: float = 0.0
    # Sprint 6 v6: scored top-K attention (avg_k ≈ fixed_k regardless of n)
    scored_topk_attn_ms: float = 0.0
    scored_topk_build_ms: float = 0.0
    # Sprint 6 v6: amortized cost (topology paid once, attention reused N times)
    amortized_cached_ms_10: float = 0.0
    amortized_cached_ms_100: float = 0.0
    # Block-level
    full_block_ms: float = 0.0
    dense_masked_block_ms: float = 0.0
    sparse_block_uncached_ms: float = 0.0
    sparse_block_cached_ms: float = 0.0
    # End-to-end
    full_e2e_ms: float = 0.0
    masked_e2e_ms: float = 0.0
    sparse_e2e_uncached_ms: float = 0.0
    sparse_e2e_cached_ms: float = 0.0
    # Relation diagnostics
    by_relation: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"n={self.n}  mode={self.node_mode}  k={self.k}  device={self.device}",
            f"edges: full={self.full_edges}  allowed={self.allowed_edges}",
            f"avg_k={self.avg_k:.2f}  max_k={self.max_k}  padding={self.padding_ratio:.3f}",
            f"sparsity={self.sparsity_ratio:.4f}  rel_reduce={self.relation_reduction:.4f}",
            "--- topology build ---",
            f"  build_ms={self.topology_build_ms:.3f}ms",
            "--- attention only ---",
            f"  dense_full={self.dense_full_attn_ms:.3f}ms  "
            f"dense_masked={self.dense_masked_attn_ms:.3f}ms  "
            f"nbr_exact={self.nbr_sparse_exact_ms:.3f}ms  "
            f"nbr_trunc={self.nbr_sparse_trunc_ms:.3f}ms  "
            f"compiled={self.compiled_sparse_attn_ms:.3f}ms  "
            f"triton={self.triton_sparse_attn_ms:.3f}ms",
            "--- block level ---",
            f"  full={self.full_block_ms:.3f}ms  "
            f"dense_masked={self.dense_masked_block_ms:.3f}ms  "
            f"sparse_uncached={self.sparse_block_uncached_ms:.3f}ms  "
            f"sparse_cached={self.sparse_block_cached_ms:.3f}ms",
            "--- end to end ---",
            f"  full={self.full_e2e_ms:.3f}ms  "
            f"masked={self.masked_e2e_ms:.3f}ms  "
            f"sparse_uncached={self.sparse_e2e_uncached_ms:.3f}ms  "
            f"sparse_cached={self.sparse_e2e_cached_ms:.3f}ms",
        ]
        if self.by_relation:
            lines.append("--- relations ---")
            for rel, cnt in self.by_relation.items():
                lines.append(f"  {rel}: {cnt}")
        return "\n".join(lines)


# ── Node collection helpers ───────────────────────────────────────────────────

def _collect_nodes_trees(exprs: list[str], target_n: int):
    from .parser import parse
    from .normalize import normalize
    nodes = []
    for expr in exprs:
        root = normalize(parse(expr))
        nodes.extend(root.collect_nodes())
        if len(nodes) >= target_n:
            break
    while len(nodes) < target_n:
        nodes.extend(nodes[:target_n - len(nodes)])
    return nodes[:target_n]


def _collect_nodes_roots(exprs: list[str], target_n: int):
    from .parser import parse
    from .normalize import normalize
    roots = [normalize(parse(e)) for e in exprs]
    nodes = []
    while len(nodes) < target_n:
        nodes.extend(roots[:target_n - len(nodes)])
    return nodes[:target_n]


def _load_env_from_examples(examples_path: str | None) -> dict[str, tuple[int, ...]]:
    if not examples_path:
        return {}
    p = Path(examples_path)
    if not p.exists():
        return {}
    with open(p) as f:
        for line in f:
            rec = json.loads(line)
            raw_shape = rec.get("shape", {})
            if raw_shape:
                return {k: tuple(v) for k, v in raw_shape.items()}
    return {}


# ── Core benchmark ────────────────────────────────────────────────────────────

def run_benchmark(
    n: int,
    node_mode: str = "roots",
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 1,
    d_ff: int = 128,
    topk: int = 3,
    local_window: int = 1,
    max_neighbors: int | None = None,
    n_warmup: int = 2,
    n_iter: int = 10,
    exprs: list[str] | None = None,
    examples_path: str | None = None,
    save_dir: str | None = None,
    topology_mode: str = "union",
    fixed_k: int = 32,
) -> BenchmarkReport:
    from .model import MathRoutedTransformer, MathRoutedTransformerBlock
    from .topology import TopologyBuilder
    from .topology_cache import TopologyCache, CachedTopology
    from .embedder import MathEmbedder
    from .sparse_attention import (
        neighbors_from_mask, neighbors_from_mask_prioritized, max_k_from_mask,
    )
    from .topology import build_priority_matrix
    from .attention import math_attention
    from .verifier import Verifier
    from .triton_attention import triton_neighbor_attention, TRITON_AVAILABLE
    import torch.nn as nn

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: CPU benchmark path has been removed")
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for neighbor-sparse attention")
    device = "cuda"

    if exprs is None:
        exprs = [
            "add(matmul(A, x), b)",
            "add(matmul(W, h), c)",
            "grad(f, x)",
            "sum(i, x_i)",
            "matmul(Q, K)",
            "constraint(leq(matmul(A, x), b))",
        ]

    env = _load_env_from_examples(examples_path)

    if node_mode == "trees":
        nodes = _collect_nodes_trees(exprs, n)
    else:
        nodes = _collect_nodes_roots(exprs, n)

    embedder = MathEmbedder()
    z = embedder.encode_batch(nodes)
    z_t = torch.from_numpy(z).float()

    tb = TopologyBuilder(topk=topk, local_window=local_window)
    np_mask, diag = tb.build_detailed(nodes, z, env or None)
    mask_t = torch.tensor(np_mask, dtype=torch.bool)

    priority = build_priority_matrix(
        nodes, z=z, env=env or None, topk=topk, local_window=local_window
    )
    exact_k = diag.max_k
    trunc_k = max_neighbors if max_neighbors is not None else max(exact_k // 2, 1)

    # Sprint 1: time the full topology build pipeline (CPU path, cache-miss cost)
    _embedder_topo = MathEmbedder()
    _tb_topo = TopologyBuilder(topk=topk, local_window=local_window)

    def _build_topo_cpu():
        _z = _embedder_topo.encode_batch(nodes)
        _mask, _diag = _tb_topo.build_detailed(nodes, _z, env or None)
        _prio = build_priority_matrix(
            nodes, z=_z, env=env or None, topk=topk, local_window=local_window,
        )
        _mt = torch.tensor(_mask, dtype=torch.bool)
        _K = max(max_neighbors if max_neighbors is not None else _diag.max_k, 1)
        neighbors_from_mask_prioritized(_mt, _prio, _K)

    topology_build_ms = _timed(_build_topo_cpu, n_warmup=1, n_iter=5)

    nb_exact, valid_exact = neighbors_from_mask_prioritized(mask_t, priority, exact_k)
    nb_trunc, valid_trunc = neighbors_from_mask_prioritized(mask_t, priority, trunc_k)

    # Move topology tensors to device
    mask_t    = mask_t.to(device)
    nb_exact  = nb_exact.to(device)
    valid_exact = valid_exact.to(device)
    nb_trunc  = nb_trunc.to(device)
    valid_trunc = valid_trunc.to(device)

    # ── Attention-only ────────────────────────────────────────────────────────
    B, H, Dh = 1, n_heads, d_model // n_heads
    q_ = torch.randn(B, H, n, Dh, device=device)
    k_ = torch.randn(B, H, n, Dh, device=device)
    v_ = torch.randn(B, H, n, Dh, device=device)
    mask_4d = mask_t.unsqueeze(0).unsqueeze(0)

    dense_full_attn_ms   = _timed(lambda: math_attention(q_, k_, v_, None), n_warmup, n_iter)
    dense_masked_attn_ms = _timed(lambda: math_attention(q_, k_, v_, mask_4d), n_warmup, n_iter)
    nbr_exact_ms = _timed(lambda: triton_neighbor_attention(q_, k_, v_, nb_exact, valid_exact), n_warmup, n_iter)
    nbr_trunc_ms = _timed(lambda: triton_neighbor_attention(q_, k_, v_, nb_trunc, valid_trunc), n_warmup, n_iter)
    compiled_ms = 0.0
    triton_ms = nbr_trunc_ms

    # v6 Sprint 1: scored top-K topology build + attention timing
    from .topology import RELATION_WEIGHTS
    scored_topk_build_ms = 0.0
    scored_topk_attn_ms = 0.0
    try:
        _tb_stk = TopologyBuilder(
            topk=topk, local_window=local_window,
            topology_mode="scored_topk", fixed_k=fixed_k,
        )
        _embedder_stk = MathEmbedder()

        def _build_scored_topk_cpu():
            _z = _embedder_stk.encode_batch(nodes)
            _mask, _diag = _tb_stk.build_scored_topk(nodes, _z, env or None)
            _mt = torch.tensor(_mask, dtype=torch.bool)
            _K = fixed_k
            neighbors_from_mask_prioritized(_mt, priority, _K)

        scored_topk_build_ms = _timed(_build_scored_topk_cpu, n_warmup=1, n_iter=5)

        # Build scored_topk neighbors for attention timing
        _z_stk = _embedder_stk.encode_batch(nodes)
        _mask_stk, _diag_stk = _tb_stk.build_scored_topk(nodes, _z_stk, env or None)
        _mask_stk_t = torch.tensor(_mask_stk, dtype=torch.bool).to(device)
        _nb_stk, _valid_stk = neighbors_from_mask_prioritized(_mask_stk_t, priority, fixed_k)
        scored_topk_attn_ms = _timed(
            lambda: triton_neighbor_attention(q_, k_, v_, _nb_stk, _valid_stk),
            n_warmup, n_iter,
        )
    except Exception:
        pass

    # v6 Sprint 3: amortized topology cost (build once, reuse N times)
    # amortized_ms = (topo_build_ms + N * attention_ms) / N
    _attn_ms = triton_ms
    amortized_cached_ms_10 = (topology_build_ms + 10 * _attn_ms) / 10
    amortized_cached_ms_100 = (topology_build_ms + 100 * _attn_ms) / 100

    # ── Block-level ───────────────────────────────────────────────────────────
    proj = nn.Linear(z_t.shape[1], d_model, bias=False).to(device)
    with torch.no_grad():
        x_dm = proj(z_t.to(device).unsqueeze(0))  # (1, T, d_model)

    block_full = MathRoutedTransformerBlock(
        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        topk=topk, local_window=local_window, attention_mode="full"
    ).to(device)
    block_masked = MathRoutedTransformerBlock(
        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        topk=topk, local_window=local_window, attention_mode="dense_masked"
    ).to(device)
    # Uncached sparse block — fresh cache each call
    block_sparse_uc = MathRoutedTransformerBlock(
        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        topk=topk, local_window=local_window, attention_mode="neighbor_sparse",
        max_neighbors=max_neighbors,
    ).to(device)
    # Cached sparse block — shared cache across calls
    shared_cache = TopologyCache()
    block_sparse_c = MathRoutedTransformerBlock(
        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        topk=topk, local_window=local_window, attention_mode="neighbor_sparse",
        max_neighbors=max_neighbors, topology_cache=shared_cache,
    ).to(device)
    # Warm the cache with one call before benchmarking
    with torch.no_grad():
        block_sparse_c(x_dm, nodes, env=env or None)

    full_block_ms         = _timed(lambda: block_full(x_dm), n_warmup, n_iter)
    dense_masked_block_ms = _timed(lambda: block_masked(x_dm, nodes, env=env or None), n_warmup, n_iter)
    sparse_uc_block_ms    = _timed(lambda: block_sparse_uc(x_dm, nodes, env=env or None), n_warmup, n_iter)
    sparse_c_block_ms     = _timed(lambda: block_sparse_c(x_dm, nodes, env=env or None), n_warmup, n_iter)

    # ── End-to-end ────────────────────────────────────────────────────────────
    verifier = Verifier()

    shared_cache_e2e = TopologyCache()
    kw = dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
              topk=topk, local_window=local_window, max_neighbors=max_neighbors)
    m_full   = MathRoutedTransformer(**kw, attention_mode="full",    share_topology_cache=False).to(device)
    m_masked = MathRoutedTransformer(**kw, attention_mode="dense_masked", share_topology_cache=False).to(device)
    m_sparse_uc = MathRoutedTransformer(**kw, attention_mode="neighbor_sparse", share_topology_cache=False).to(device)
    m_sparse_c  = MathRoutedTransformer(**kw, attention_mode="neighbor_sparse", share_topology_cache=True).to(device)

    xf    = m_full.embed_nodes(nodes).to(device)
    xm    = m_masked.embed_nodes(nodes).to(device)
    xs_uc = m_sparse_uc.embed_nodes(nodes).to(device)
    xs_c  = m_sparse_c.embed_nodes(nodes).to(device)

    # Warm e2e cached model
    with torch.no_grad():
        m_sparse_c(xs_c, nodes, env=env or None)

    def _e2e(model, x_in, n_nodes, pass_nodes):
        model(x_in, n_nodes if pass_nodes else None, env=env or None)
        for nd in nodes[:min(4, len(nodes))]:  # cap verifier calls to keep timing honest
            verifier.check_tree(nd)

    full_e2e_ms      = _timed(lambda: _e2e(m_full,      xf,   None,  False), n_warmup, n_iter)
    masked_e2e_ms    = _timed(lambda: _e2e(m_masked,    xm,   nodes, True),  n_warmup, n_iter)
    sparse_uc_e2e_ms = _timed(lambda: _e2e(m_sparse_uc, xs_uc, nodes, True),  n_warmup, n_iter)
    sparse_c_e2e_ms  = _timed(lambda: _e2e(m_sparse_c,  xs_c,  nodes, True),  n_warmup, n_iter)

    report = BenchmarkReport(
        n=n,
        node_mode=node_mode,
        k=topk,
        full_edges=n * n,
        allowed_edges=diag.allowed_edges,
        avg_k=diag.avg_k,
        max_k=diag.max_k,
        padding_ratio=diag.padding_ratio,
        sparsity_ratio=diag.sparsity_ratio,
        relation_reduction=diag.relation_reduction,
        device=device,
        topology_build_ms=topology_build_ms,
        dense_full_attn_ms=dense_full_attn_ms,
        dense_masked_attn_ms=dense_masked_attn_ms,
        nbr_sparse_exact_ms=nbr_exact_ms,
        nbr_sparse_trunc_ms=nbr_trunc_ms,
        compiled_sparse_attn_ms=compiled_ms,
        triton_sparse_attn_ms=triton_ms,
        scored_topk_attn_ms=scored_topk_attn_ms,
        scored_topk_build_ms=scored_topk_build_ms,
        amortized_cached_ms_10=amortized_cached_ms_10,
        amortized_cached_ms_100=amortized_cached_ms_100,
        full_block_ms=full_block_ms,
        dense_masked_block_ms=dense_masked_block_ms,
        sparse_block_uncached_ms=sparse_uc_block_ms,
        sparse_block_cached_ms=sparse_c_block_ms,
        full_e2e_ms=full_e2e_ms,
        masked_e2e_ms=masked_e2e_ms,
        sparse_e2e_uncached_ms=sparse_uc_e2e_ms,
        sparse_e2e_cached_ms=sparse_c_e2e_ms,
        by_relation=diag.by_relation,
    )

    if save_dir:
        _save_report(report, save_dir)

    return report


def _save_report(report: BenchmarkReport, save_dir: str) -> None:
    """Persist benchmark report to JSON under runs/benchmarks/."""
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    import datetime
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    fname = p / f"bench_n{report.n}_{report.node_mode}_{ts}.json"
    data = asdict(report)
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)


# ── Accuracy metrics ──────────────────────────────────────────────────────────

def op_accuracy(predictions: list[str], targets: list[str]) -> float:
    if not targets:
        return 0.0
    return sum(p == t for p, t in zip(predictions, targets)) / len(targets)


def route_accuracy(routes: list[str], targets: list[str]) -> float:
    return op_accuracy(routes, targets)


def retrieval_recall_at_k(
    retrieved: list[int], relevant: set[int], k: int
) -> float:
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant) if relevant else 0.0


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_table_header() -> None:
    cols = (
        f"{'n':>5}  {'mode':>5}  {'allowed':>8}  {'full':>6}  "
        f"{'avg_k':>6}  {'max_k':>5}  {'rel_red':>7}  "
        f"{'topo_ms':>8}  "
        f"{'d_attn':>7}  {'s_trnc':>7}  {'s_comp':>7}  {'s_tri':>7}  "
        f"{'stk_bld':>8}  {'stk_atn':>8}  "
        f"{'amrt_10':>8}  {'amrt_100':>9}  "
        f"{'d_blk':>7}  {'s_uc':>7}  {'s_c':>7}"
    )
    print(cols)
    print("-" * len(cols))


def _print_row(r: BenchmarkReport) -> None:
    print(
        f"{r.n:>5}  {r.node_mode:>5}  {r.allowed_edges:>8}  {r.full_edges:>6}  "
        f"{r.avg_k:>6.1f}  {r.max_k:>5}  {r.relation_reduction:>7.4f}  "
        f"{r.topology_build_ms:>8.3f}  "
        f"{r.dense_full_attn_ms:>7.3f}  {r.nbr_sparse_trunc_ms:>7.3f}  "
        f"{r.compiled_sparse_attn_ms:>7.3f}  {r.triton_sparse_attn_ms:>7.3f}  "
        f"{r.scored_topk_build_ms:>8.3f}  {r.scored_topk_attn_ms:>8.3f}  "
        f"{r.amortized_cached_ms_10:>8.3f}  {r.amortized_cached_ms_100:>9.3f}  "
        f"{r.full_block_ms:>7.3f}  {r.sparse_block_uncached_ms:>7.3f}  "
        f"{r.sparse_block_cached_ms:>7.3f}"
    )
    if r.by_relation:
        for rel, cnt in r.by_relation.items():
            print(f"         {rel}: {cnt}")


def _run_benchmark_cli(args: argparse.Namespace) -> None:
    sizes = [int(x) for x in args.sizes.split(",")]
    modes = args.node_mode.split(",")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: CPU benchmark path has been removed")
    print(f"device={torch.device('cuda')}  "
          f"threads={torch.get_num_threads()}  python={sys.version.split()[0]}")
    _print_table_header()

    save_dir = args.save_dir if hasattr(args, "save_dir") else None

    for mode in modes:
        for n in sizes:
            r = run_benchmark(
                n=n,
                node_mode=mode,
                d_model=args.d_model,
                n_heads=args.n_heads,
                topk=args.topk,
                local_window=args.local_window,
                max_neighbors=args.max_neighbors,
                n_warmup=args.warmup,
                n_iter=args.iters,
                examples_path=args.examples,
                save_dir=save_dir,
                topology_mode=args.topology_mode,
                fixed_k=args.fixed_k,
            )
            _print_row(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval / benchmark Math-Routed Transformer")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--sizes", default="8,16,32")
    parser.add_argument("--node-mode", default="roots,trees", dest="node_mode")
    parser.add_argument("--d-model", type=int, default=64, dest="d_model")
    parser.add_argument("--n-heads", type=int, default=4, dest="n_heads")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--local-window", type=int, default=1, dest="local_window")
    parser.add_argument("--max-neighbors", type=int, default=32, dest="max_neighbors")
    parser.add_argument("--topology-mode", default="union", dest="topology_mode",
                        choices=["union", "scored_topk"])
    parser.add_argument("--fixed-k", type=int, default=32, dest="fixed_k")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--examples", default=None)
    parser.add_argument("--save-dir", default=None, dest="save_dir")
    args = parser.parse_args()
    if args.benchmark:
        _run_benchmark_cli(args)


if __name__ == "__main__":
    main()
