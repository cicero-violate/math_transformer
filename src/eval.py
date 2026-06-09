from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch

from .model import DEFAULT_MAX_NEIGHBORS


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
    selector_results: dict[str, dict[str, float]] = field(default_factory=dict)
    # Sprint 6 v6: amortized cost (topology paid once, attention reused N times)
    amortized_cached_ms_10: float = 0.0
    amortized_cached_ms_100: float = 0.0
    # Block-level
    full_block_ms: float = 0.0
    dense_masked_block_ms: float = 0.0
    sparse_block_uncached_ms: float = 0.0
    sparse_block_cached_ms: float = 0.0
    prepared_static_sparse_block_ms: float = 0.0
    prepared_static_sparse_attention_ms: float = 0.0
    prepared_static_sparse_non_attention_ms: float = 0.0
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
        ]
        if self.selector_results:
            lines.append("--- selector comparison ---")
            for mode, vals in self.selector_results.items():
                lines.append(
                    f"  {mode}: attn={vals.get('attn_ms', 0.0):.3f}ms  "
                    f"block={vals.get('cached_block_ms', 0.0):.3f}ms  "
                    f"dense_proxy_l1={vals.get('dense_proxy_l1', 0.0):.6f}  "
                    f"dense_proxy_cos={vals.get('dense_proxy_cos', 0.0):.6f}"
                )
        lines.extend([
            "--- block level ---",
            f"  full={self.full_block_ms:.3f}ms  "
            f"dense_masked={self.dense_masked_block_ms:.3f}ms  "
            f"sparse_uncached={self.sparse_block_uncached_ms:.3f}ms  "
            f"sparse_cached={self.sparse_block_cached_ms:.3f}ms  "
            f"prepared_static={self.prepared_static_sparse_block_ms:.3f}ms  "
            f"prepared_attn={self.prepared_static_sparse_attention_ms:.3f}ms  "
            f"prepared_non_attn={self.prepared_static_sparse_non_attention_ms:.3f}ms",
            "--- end to end ---",
            f"  full={self.full_e2e_ms:.3f}ms  "
            f"masked={self.masked_e2e_ms:.3f}ms  "
            f"sparse_uncached={self.sparse_e2e_uncached_ms:.3f}ms  "
            f"sparse_cached={self.sparse_e2e_cached_ms:.3f}ms",
        ])
        if self.by_relation:
            lines.append("--- relations ---")
            for rel, cnt in self.by_relation.items():
                lines.append(f"  {rel}: {cnt}")
        return "\n".join(lines)


@dataclass
class QualityReport:
    mode: str
    k: int | None
    n_examples: int
    route_accuracy: float
    dense_agreement: float | None = None
    hidden_l1: float | None = None
    hidden_cos: float | None = None
    logit_l1: float | None = None
    logit_kl_dense_to_sparse: float | None = None
    by_expert: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def __str__(self) -> str:
        k_str = "full" if self.k is None else str(self.k)
        agree = "" if self.dense_agreement is None else f"  dense_agree={self.dense_agreement:.4f}"
        hidden = ""
        if self.hidden_l1 is not None:
            hidden += f"  hidden_l1={self.hidden_l1:.6f}"
        if self.hidden_cos is not None:
            hidden += f"  hidden_cos={self.hidden_cos:.6f}"
        logits = ""
        if self.logit_l1 is not None:
            logits += f"  logit_l1={self.logit_l1:.6f}"
        if self.logit_kl_dense_to_sparse is not None:
            logits += f"  logit_kl={self.logit_kl_dense_to_sparse:.6f}"
        line = (
            f"mode={self.mode}  k={k_str}  examples={self.n_examples}  "
            f"route_acc={self.route_accuracy:.4f}{agree}{hidden}{logits}"
        )
        if not self.by_expert:
            return line
        details = "  ".join(
            f"{expert}={stats['correct']}/{stats['total']}({stats['accuracy']:.4f})"
            for expert, stats in sorted(self.by_expert.items())
        )
        return f"{line}\n         by_expert {details}"


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


def _load_route_eval_records(examples_path: str) -> list[dict]:
    from .tasks import EXPERT_TO_ID

    records: list[dict] = []
    with open(examples_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            expert = rec.get("expert")
            if expert not in EXPERT_TO_ID:
                continue
            raw_shape = rec.get("shape") or {}
            env = {
                k: tuple(v)
                for k, v in raw_shape.items()
                if k != "out" and isinstance(v, list)
            }
            records.append({
                "expr": rec.get("normalized") or rec.get("expr", ""),
                "expert_id": EXPERT_TO_ID[expert],
                "expert": expert,
                "env": env,
            })
    return records


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
    max_neighbors: int | None = DEFAULT_MAX_NEIGHBORS,
    n_warmup: int = 2,
    n_iter: int = 10,
    exprs: list[str] | None = None,
    examples_path: str | None = None,
    save_dir: str | None = None,
    topology_mode: str = "union",
    fixed_k: int = 32,
    middle_bridge_width: int = 0,
    selector_alpha: float = 1.0,
    selector_beta: float = 1.0,
    selector_candidate_neighbors: int | None = None,
    profile_prepared_block: bool = False,
    learned_scorer_checkpoint: str | None = None,
    learned_k: int = 8,
) -> BenchmarkReport:
    from .model import MathRoutedTransformer, MathRoutedTransformerBlock
    from .topology import TopologyBuilder
    from .learned_topology_runtime import LearnedTopologyBuilder
    from .topology_cache import TopologyCache, CachedTopology
    from .embedder import MathEmbedder
    from .sparse_attention import (
        neighbors_from_mask, neighbors_from_mask_prioritized, max_k_from_mask,
        neighbors_from_candidate_qk_scores, neighbors_from_qk_scores,
        symbolic_priority_scores,
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

    def _make_builder():
        if topology_mode == "learned_topology":
            if not learned_scorer_checkpoint:
                raise ValueError("--learned-scorer-checkpoint is required for learned_topology benchmark")
            return LearnedTopologyBuilder(
                learned_scorer_checkpoint,
                fixed_k=learned_k,
                topk=topk,
                local_window=local_window,
                middle_bridge_width=middle_bridge_width,
                device=torch.device(device),
            )
        return TopologyBuilder(
            topk=topk,
            local_window=local_window,
            topology_mode=topology_mode,
            fixed_k=fixed_k,
            middle_bridge_width=middle_bridge_width,
        )

    tb = _make_builder()

    def _build_mask(builder, z_arr: np.ndarray):
        if getattr(builder, "topology_mode", "union") in ("scored_topk", "middle_preserving_topk", "learned_topology"):
            return builder.build_scored_topk(nodes, z_arr, env or None)
        return builder.build_detailed(nodes, z_arr, env or None)

    np_mask, diag = _build_mask(tb, z)
    mask_t = torch.tensor(np_mask, dtype=torch.bool)

    if getattr(tb, "is_learned_topology", False):
        priority = tb.priority_from_mask(np_mask)
    else:
        priority = build_priority_matrix(
            nodes,
            z=z,
            env=env or None,
            topk=topk,
            local_window=local_window,
            include_middle_bridge=tb.include_middle_bridge,
            middle_bridge_width=middle_bridge_width,
        )
    exact_k = diag.max_k
    trunc_k = max_neighbors if max_neighbors is not None else max(exact_k // 2, 1)
    candidate_k = selector_candidate_neighbors
    if candidate_k is None:
        candidate_k = min(max(trunc_k * 4, trunc_k), exact_k)
    candidate_k = max(candidate_k, trunc_k, 1)

    # Sprint 1: time the full topology build pipeline (CPU path, cache-miss cost)
    _embedder_topo = MathEmbedder()
    _tb_topo = _make_builder()

    def _build_topo_cpu():
        _z = _embedder_topo.encode_batch(nodes)
        _mask, _diag = _build_mask(_tb_topo, _z)
        if getattr(_tb_topo, "is_learned_topology", False):
            _prio = _tb_topo.priority_from_mask(_mask)
        else:
            _prio = build_priority_matrix(
                nodes,
                z=_z,
                env=env or None,
                topk=topk,
                local_window=local_window,
                include_middle_bridge=_tb_topo.include_middle_bridge,
                middle_bridge_width=middle_bridge_width,
            )
        _mt = torch.tensor(_mask, dtype=torch.bool)
        _K = max(max_neighbors if max_neighbors is not None else _diag.max_k, 1)
        neighbors_from_mask_prioritized(_mt, _prio, _K)

    topology_build_ms = _timed(_build_topo_cpu, n_warmup=1, n_iter=5)

    nb_exact, valid_exact = neighbors_from_mask_prioritized(mask_t, priority, exact_k)
    nb_trunc, valid_trunc = neighbors_from_mask_prioritized(mask_t, priority, trunc_k)
    nb_candidate, valid_candidate = neighbors_from_mask_prioritized(mask_t, priority, candidate_k)

    # Move topology tensors to device
    mask_t    = mask_t.to(device)
    nb_exact  = nb_exact.to(device)
    valid_exact = valid_exact.to(device)
    nb_trunc  = nb_trunc.to(device)
    valid_trunc = valid_trunc.to(device)
    nb_candidate = nb_candidate.to(device)
    valid_candidate = valid_candidate.to(device)

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
    scored_topk_build_ms = 0.0
    scored_topk_attn_ms = 0.0
    _tb_stk = TopologyBuilder(
        topk=topk, local_window=local_window,
        topology_mode="scored_topk", fixed_k=fixed_k,
    )
    _tb_v7 = TopologyBuilder(
        topk=topk, local_window=local_window,
        topology_mode="middle_preserving_topk", fixed_k=fixed_k,
        middle_bridge_width=middle_bridge_width,
    )
    _embedder_stk = MathEmbedder()

    def _build_scored_topk_cpu():
        _z = _embedder_stk.encode_batch(nodes)
        _mask, _diag = _tb_stk.build_scored_topk(nodes, _z, env or None)
        _mt = torch.tensor(_mask, dtype=torch.bool)
        neighbors_from_mask(_mt, fixed_k)

    scored_topk_build_ms = _timed(_build_scored_topk_cpu, n_warmup=1, n_iter=5)

    # Build scored_topk neighbors for attention timing.
    _z_stk = _embedder_stk.encode_batch(nodes)
    _mask_stk, _diag_stk = _tb_stk.build_scored_topk(nodes, _z_stk, env or None)
    _mask_stk_t = torch.tensor(_mask_stk, dtype=torch.bool)
    _nb_stk, _valid_stk = neighbors_from_mask(_mask_stk_t, fixed_k)
    _nb_stk = _nb_stk.to(device)
    _valid_stk = _valid_stk.to(device)
    scored_topk_attn_ms = _timed(
        lambda: triton_neighbor_attention(q_, k_, v_, _nb_stk, _valid_stk),
        n_warmup, n_iter,
    )

    priority_t = torch.tensor(priority, dtype=torch.int8, device=device)
    symbolic_scores_t = symbolic_priority_scores(priority_t)

    def _attn_for_selector(selector: str) -> torch.Tensor:
        if selector == "topology_only":
            return triton_neighbor_attention(q_, k_, v_, nb_trunc, valid_trunc)
        if selector == "kmip_only":
            nb_sel, valid_sel = neighbors_from_qk_scores(q_, k_, trunc_k)
        elif selector == "symbolic_kmip":
            nb_sel, valid_sel = neighbors_from_qk_scores(
                q_, k_, trunc_k,
                symbolic_scores=symbolic_scores_t,
                alpha=selector_alpha,
                beta=selector_beta,
            )
        elif selector == "symbolic_candidate_kmip":
            nb_sel, valid_sel = neighbors_from_candidate_qk_scores(
                q_, k_,
                candidate_neighbors=nb_candidate,
                candidate_valid=valid_candidate,
                max_k=trunc_k,
                symbolic_scores=symbolic_scores_t,
                alpha=selector_alpha,
                beta=selector_beta,
            )
        else:
            raise ValueError(f"unknown selector: {selector}")
        return triton_neighbor_attention(q_, k_, v_, nb_sel, valid_sel)

    with torch.no_grad():
        dense_ref = math_attention(q_, k_, v_, None)

    selector_results: dict[str, dict[str, float]] = {}
    for selector in ("topology_only", "kmip_only", "symbolic_kmip", "symbolic_candidate_kmip"):
        attn_ms = triton_ms if selector == "topology_only" else _timed(
            lambda s=selector: _attn_for_selector(s), n_warmup, n_iter
        )
        with torch.no_grad():
            out_sel = _attn_for_selector(selector)
            dense_proxy_l1 = (out_sel - dense_ref).abs().mean().item()
            dense_proxy_cos = torch.nn.functional.cosine_similarity(
                out_sel.reshape(1, -1),
                dense_ref.reshape(1, -1),
                dim=1,
            ).item()
        selector_results[selector] = {
            "attn_ms": float(attn_ms),
            "dense_proxy_l1": float(dense_proxy_l1),
            "dense_proxy_cos": float(dense_proxy_cos),
        }

    # v6 Sprint 3: amortized topology cost (build once, reuse N times)
    # amortized_ms = (topo_build_ms + N * attention_ms) / N
    _attn_ms = triton_ms
    amortized_cached_ms_10 = (topology_build_ms + 10 * _attn_ms) / 10
    amortized_cached_ms_100 = (topology_build_ms + 100 * _attn_ms) / 100

    def _install_learned_topology(block):
        if topology_mode == "learned_topology":
            block.topology = _make_builder()
            block.max_neighbors = learned_k
        return block

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
        topology_mode=topology_mode,
        fixed_k=fixed_k,
        middle_bridge_width=middle_bridge_width,
    ).to(device)
    _install_learned_topology(block_sparse_uc)
    # Cached sparse block — shared cache across calls
    shared_cache = TopologyCache()
    block_sparse_c = MathRoutedTransformerBlock(
        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        topk=topk, local_window=local_window, attention_mode="neighbor_sparse",
        max_neighbors=max_neighbors, topology_cache=shared_cache,
        topology_mode=topology_mode,
        fixed_k=fixed_k,
        middle_bridge_width=middle_bridge_width,
    ).to(device)
    _install_learned_topology(block_sparse_c)
    # Warm the cache with one call before benchmarking
    with torch.no_grad():
        block_sparse_c(x_dm, nodes, env=env or None, return_metadata=False)

    full_block_ms         = _timed(lambda: block_full(x_dm), n_warmup, n_iter)
    dense_masked_block_ms = _timed(lambda: block_masked(x_dm, nodes, env=env or None), n_warmup, n_iter)
    sparse_uc_block_ms    = _timed(lambda: block_sparse_uc(x_dm, nodes, env=env or None), n_warmup, n_iter)
    sparse_c_block_ms     = _timed(lambda: block_sparse_c(x_dm, nodes, env=env or None, return_metadata=False), n_warmup, n_iter)
    selector_results["topology_only"]["cached_block_ms"] = float(sparse_c_block_ms)
    prepared_static_sparse_block_ms = 0.0
    prepared_static_sparse_attention_ms = 0.0
    prepared_static_sparse_non_attention_ms = 0.0
    if profile_prepared_block:
        prepared_cache = TopologyCache()
        block_prepared = MathRoutedTransformerBlock(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            topk=topk, local_window=local_window, attention_mode="neighbor_sparse",
            max_neighbors=max_neighbors, topology_cache=prepared_cache,
            topology_mode=topology_mode,
            fixed_k=fixed_k,
            middle_bridge_width=middle_bridge_width,
        ).to(device)
        _install_learned_topology(block_prepared)
        block_prepared.eval()
        with torch.no_grad():
            block_prepared.prepare_static_topology(nodes, env=env or None, device=torch.device(device))
            block_prepared.profile_cached_sparse_block(x_dm, nodes, env=env or None)

        def _profile_prepared_once() -> dict[str, float]:
            with torch.no_grad():
                _, timings = block_prepared.profile_cached_sparse_block(x_dm, nodes, env=env or None)
            return timings

        prepared_samples = []
        with torch.no_grad():
            for _ in range(n_warmup):
                _profile_prepared_once()
            for _ in range(n_iter):
                prepared_samples.append(_profile_prepared_once())

        prepared_totals = np.array([t.get("total_block_ms", 0.0) for t in prepared_samples])
        prepared_attns = np.array([t.get("attention_kernel_ms", 0.0) for t in prepared_samples])
        prepared_static_sparse_block_ms = float(np.median(prepared_totals))
        prepared_static_sparse_attention_ms = float(np.median(prepared_attns))
        prepared_static_sparse_non_attention_ms = max(
            0.0,
            prepared_static_sparse_block_ms - prepared_static_sparse_attention_ms,
        )

    for selector in ("kmip_only", "symbolic_kmip", "symbolic_candidate_kmip"):
        selector_cache = TopologyCache()
        selector_cache_k = candidate_k if selector == "symbolic_candidate_kmip" else max_neighbors
        block_selector = MathRoutedTransformerBlock(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            topk=topk, local_window=local_window, attention_mode="neighbor_sparse",
            max_neighbors=selector_cache_k, topology_cache=selector_cache,
            topology_mode=topology_mode,
            fixed_k=fixed_k,
            middle_bridge_width=middle_bridge_width,
            sparse_selector=selector,
            selector_alpha=selector_alpha,
            selector_beta=selector_beta,
            selector_k=trunc_k if selector == "symbolic_candidate_kmip" else None,
        ).to(device)
        _install_learned_topology(block_selector)
        with torch.no_grad():
            block_selector(x_dm, nodes, env=env or None, return_metadata=False)
        selector_results[selector]["cached_block_ms"] = float(
            _timed(
                lambda b=block_selector: b(x_dm, nodes, env=env or None, return_metadata=False),
                n_warmup, n_iter,
            )
        )

    # ── End-to-end ────────────────────────────────────────────────────────────
    verifier = Verifier()

    shared_cache_e2e = TopologyCache()
    kw = dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers, d_ff=d_ff,
              topk=topk, local_window=local_window, max_neighbors=max_neighbors,
              topology_mode=topology_mode, fixed_k=fixed_k,
              middle_bridge_width=middle_bridge_width)
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
        selector_results=selector_results,
        amortized_cached_ms_10=amortized_cached_ms_10,
        amortized_cached_ms_100=amortized_cached_ms_100,
        full_block_ms=full_block_ms,
        dense_masked_block_ms=dense_masked_block_ms,
        sparse_block_uncached_ms=sparse_uc_block_ms,
        sparse_block_cached_ms=sparse_c_block_ms,
        prepared_static_sparse_block_ms=prepared_static_sparse_block_ms,
        prepared_static_sparse_attention_ms=prepared_static_sparse_attention_ms,
        prepared_static_sparse_non_attention_ms=prepared_static_sparse_non_attention_ms,
        full_e2e_ms=full_e2e_ms,
        masked_e2e_ms=masked_e2e_ms,
        sparse_e2e_uncached_ms=sparse_uc_e2e_ms,
        sparse_e2e_cached_ms=sparse_c_e2e_ms,
        by_relation=diag.by_relation,
    )

    if save_dir:
        _save_report(report, save_dir)

    return report


def run_quality_eval(
    examples_path: str,
    k_values: list[int],
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 128,
    topk: int = 3,
    local_window: int = 1,
    checkpoint: str | None = None,
    device: str | None = None,
    topology_mode: str = "union",
    fixed_k: int = 32,
    middle_bridge_width: int = 0,
    relation_weights: dict[str, float] | None = None,
    learned_scorer_checkpoint: str | None = None,
    learned_k: int = 8,
) -> list[QualityReport]:
    from .model import MathRoutedTransformer
    from .parser import parse
    from .normalize import normalize
    from .learned_topology_runtime import LearnedTopologyBuilder

    records = _load_route_eval_records(examples_path)
    if not records:
        raise ValueError(f"no route examples found in {examples_path}")

    if device in (None, "auto"):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for quality eval but CUDA is not available")
    torch.manual_seed(0)
    base_kwargs = dict(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=0.0,
        topk=topk,
        local_window=local_window,
        topology_mode=topology_mode,
        fixed_k=fixed_k,
        middle_bridge_width=middle_bridge_width,
        relation_weights=relation_weights,
    )
    dense = MathRoutedTransformer(
        **base_kwargs,
        attention_mode="full",
        share_topology_cache=False,
    ).to(dev)

    if checkpoint:
        try:
            state = torch.load(checkpoint, map_location=dev, weights_only=True)
        except TypeError:  # older torch
            state = torch.load(checkpoint, map_location=dev)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        dense.load_state_dict(state)
    dense.eval()
    dense_state = dense.state_dict()

    def _forward_records(
        model: MathRoutedTransformer,
        pass_nodes: bool,
    ) -> tuple[list[int], list[torch.Tensor], list[torch.Tensor]]:
        preds: list[int] = []
        hiddens: list[torch.Tensor] = []
        logits_list: list[torch.Tensor] = []
        model.eval()
        with torch.no_grad():
            for rec in records:
                root = normalize(parse(rec["expr"]))
                nodes = root.collect_nodes()
                x = model.embed_nodes(nodes).to(dev)
                out = model(x, nodes if pass_nodes else None, env=rec["env"] or None)[0]
                logits = model.route_logits(out)
                preds.append(int(logits.argmax(dim=-1).item()))
                hiddens.append(out.detach().float().cpu())
                logits_list.append(logits.detach().float().cpu())
        return preds, hiddens, logits_list

    def _agreement_metrics(
        sparse_hiddens: list[torch.Tensor],
        sparse_logits: list[torch.Tensor],
        dense_hiddens: list[torch.Tensor],
        dense_logits: list[torch.Tensor],
    ) -> dict[str, float]:
        if not dense_hiddens:
            return {
                "hidden_l1": 0.0,
                "hidden_cos": 0.0,
                "logit_l1": 0.0,
                "logit_kl_dense_to_sparse": 0.0,
            }
        hidden_l1_vals: list[float] = []
        hidden_cos_vals: list[float] = []
        logit_l1_vals: list[float] = []
        logit_kl_vals: list[float] = []
        for hs, ls, hd, ld in zip(sparse_hiddens, sparse_logits, dense_hiddens, dense_logits):
            hidden_l1_vals.append(float((hs - hd).abs().mean().item()))
            hidden_cos_vals.append(float(torch.nn.functional.cosine_similarity(
                hs.reshape(1, -1), hd.reshape(1, -1), dim=1
            ).item()))
            logit_l1_vals.append(float((ls - ld).abs().mean().item()))
            logit_kl_vals.append(float(torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(ls, dim=-1),
                torch.nn.functional.softmax(ld, dim=-1),
                reduction="batchmean",
            ).item()))
        return {
            "hidden_l1": float(np.mean(hidden_l1_vals)),
            "hidden_cos": float(np.mean(hidden_cos_vals)),
            "logit_l1": float(np.mean(logit_l1_vals)),
            "logit_kl_dense_to_sparse": float(np.mean(logit_kl_vals)),
        }

    targets = [int(rec["expert_id"]) for rec in records]
    expert_names = [str(rec["expert"]) for rec in records]

    def _by_expert(preds: list[int]) -> dict[str, dict[str, float | int]]:
        stats: dict[str, dict[str, float | int]] = {}
        for pred, target, expert in zip(preds, targets, expert_names):
            entry = stats.setdefault(expert, {"correct": 0, "total": 0, "accuracy": 0.0})
            entry["total"] = int(entry["total"]) + 1
            if pred == target:
                entry["correct"] = int(entry["correct"]) + 1
        for entry in stats.values():
            total = int(entry["total"])
            correct = int(entry["correct"])
            entry["accuracy"] = correct / total if total else 0.0
        return stats

    dense_preds, dense_hiddens, dense_logits = _forward_records(dense, pass_nodes=False)
    reports = [
        QualityReport(
            mode="full",
            k=None,
            n_examples=len(records),
            route_accuracy=op_accuracy(dense_preds, targets),
            dense_agreement=None,
            by_expert=_by_expert(dense_preds),
        )
    ]

    for k in k_values:
        sparse = MathRoutedTransformer(
            **base_kwargs,
            attention_mode="neighbor_sparse",
            max_neighbors=k,
            share_topology_cache=True,
            sparse_selector="topology_only",
        ).to(dev)
        sparse.load_state_dict(dense_state)
        sparse_preds, sparse_hiddens, sparse_logits = _forward_records(sparse, pass_nodes=True)
        agreement = _agreement_metrics(
            sparse_hiddens=sparse_hiddens,
            sparse_logits=sparse_logits,
            dense_hiddens=dense_hiddens,
            dense_logits=dense_logits,
        )
        reports.append(
            QualityReport(
                mode="topology_only",
                k=k,
                n_examples=len(records),
                route_accuracy=op_accuracy(sparse_preds, targets),
                dense_agreement=op_accuracy(sparse_preds, dense_preds),
                hidden_l1=agreement["hidden_l1"],
                hidden_cos=agreement["hidden_cos"],
                logit_l1=agreement["logit_l1"],
                logit_kl_dense_to_sparse=agreement["logit_kl_dense_to_sparse"],
                by_expert=_by_expert(sparse_preds),
            )
        )

    if learned_scorer_checkpoint:
        learned = MathRoutedTransformer(
            **base_kwargs,
            attention_mode="neighbor_sparse",
            max_neighbors=learned_k,
            share_topology_cache=True,
            sparse_selector="topology_only",
        ).to(dev)
        learned.load_state_dict(dense_state)
        for layer in learned.layers:
            layer.topology = LearnedTopologyBuilder(
                learned_scorer_checkpoint,
                fixed_k=learned_k,
                topk=topk,
                local_window=local_window,
                middle_bridge_width=middle_bridge_width,
                device=dev,
            )
            layer.max_neighbors = learned_k
        learned_preds, learned_hiddens, learned_logits = _forward_records(learned, pass_nodes=True)
        agreement = _agreement_metrics(
            sparse_hiddens=learned_hiddens,
            sparse_logits=learned_logits,
            dense_hiddens=dense_hiddens,
            dense_logits=dense_logits,
        )
        reports.append(
            QualityReport(
                mode="learned_topology",
                k=learned_k,
                n_examples=len(records),
                route_accuracy=op_accuracy(learned_preds, targets),
                dense_agreement=op_accuracy(learned_preds, dense_preds),
                hidden_l1=agreement["hidden_l1"],
                hidden_cos=agreement["hidden_cos"],
                logit_l1=agreement["logit_l1"],
                logit_kl_dense_to_sparse=agreement["logit_kl_dense_to_sparse"],
                by_expert=_by_expert(learned_preds),
            )
        )

    return reports


def _save_report(report: BenchmarkReport, save_dir: str) -> None:
    """Persist benchmark report to JSON under runs/benchmarks/."""
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    import datetime
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S")
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
        f"{'d_blk':>7}  {'s_uc':>7}  {'s_c':>7}  {'p_blk':>7}  {'p_attn':>7}  {'p_non':>7}"
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
        f"{r.sparse_block_cached_ms:>7.3f}  "
        f"{r.prepared_static_sparse_block_ms:>7.3f}  "
        f"{r.prepared_static_sparse_attention_ms:>7.3f}  "
        f"{r.prepared_static_sparse_non_attention_ms:>7.3f}"
    )
    if r.by_relation:
        for rel, cnt in r.by_relation.items():
            print(f"         {rel}: {cnt}")
    if r.selector_results:
        for mode, vals in r.selector_results.items():
            print(
                f"         selector={mode} "
                f"attn={vals.get('attn_ms', 0.0):.3f}ms "
                f"block={vals.get('cached_block_ms', 0.0):.3f}ms "
                f"dense_l1={vals.get('dense_proxy_l1', 0.0):.6f} "
                f"dense_cos={vals.get('dense_proxy_cos', 0.0):.6f}"
            )


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
                middle_bridge_width=args.middle_bridge_width,
                selector_alpha=args.selector_alpha,
                selector_beta=args.selector_beta,
                selector_candidate_neighbors=args.selector_candidate_neighbors,
                profile_prepared_block=args.profile_prepared_block,
                learned_scorer_checkpoint=args.learned_scorer_checkpoint,
                learned_k=args.learned_k,
            )
            _print_row(r)


def _run_quality_cli(args: argparse.Namespace) -> None:
    k_values = [int(x) for x in args.quality_k.split(",") if x.strip()]
    relation_weights = None
    if args.relation_weights_json:
        relation_weights = json.loads(Path(args.relation_weights_json).read_text())
    reports = run_quality_eval(
        examples_path=args.examples,
        k_values=k_values,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        topk=args.topk,
        local_window=args.local_window,
        checkpoint=args.checkpoint,
        device=args.quality_device,
        topology_mode=args.topology_mode,
        fixed_k=args.fixed_k,
        middle_bridge_width=args.middle_bridge_width,
        relation_weights=relation_weights,
        learned_scorer_checkpoint=args.learned_scorer_checkpoint,
        learned_k=args.learned_k,
    )
    print(f"examples={args.examples}  checkpoint={args.checkpoint or 'random_init'}")
    for report in reports:
        print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval / benchmark Math-Routed Transformer")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument('--profile-prepared-block', action="store_true", dest="profile_prepared_block")
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--sizes", default="8,16,32")
    parser.add_argument("--node-mode", default="roots,trees", dest="node_mode")
    parser.add_argument("--d-model", type=int, default=64, dest="d_model")
    parser.add_argument("--n-heads", type=int, default=4, dest="n_heads")
    parser.add_argument("--n-layers", type=int, default=2, dest="n_layers")
    parser.add_argument("--d-ff", type=int, default=128, dest="d_ff")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--local-window", type=int, default=1, dest="local_window")
    parser.add_argument("--max-neighbors", type=int, default=DEFAULT_MAX_NEIGHBORS, dest="max_neighbors")
    parser.add_argument("--topology-mode", default="union", dest="topology_mode",
                        choices=["union", "scored_topk", "middle_preserving_topk", "learned_topology"])
    parser.add_argument("--fixed-k", type=int, default=32, dest="fixed_k")
    parser.add_argument("--middle-bridge-width", type=int, default=0, dest="middle_bridge_width")
    parser.add_argument("--selector-alpha", type=float, default=1.0, dest="selector_alpha")
    parser.add_argument("--selector-beta", type=float, default=1.0, dest="selector_beta")
    parser.add_argument(
        "--selector-candidate-neighbors",
        type=int,
        default=None,
        dest="selector_candidate_neighbors",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--examples", default="data/examples.jsonl")
    parser.add_argument("--quality-k", default="16,32,64,128", dest="quality_k")
    parser.add_argument("--quality-device", default=None, dest="quality_device")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--relation-weights-json", default=None, dest="relation_weights_json")
    parser.add_argument("--learned-scorer-checkpoint", default=None, dest="learned_scorer_checkpoint")
    parser.add_argument("--learned-k", type=int, default=8, dest="learned_k")
    parser.add_argument("--save-dir", default=None, dest="save_dir")
    args = parser.parse_args()
    if args.benchmark:
        _run_benchmark_cli(args)
    if args.quality:
        _run_quality_cli(args)


if __name__ == "__main__":
    main()
