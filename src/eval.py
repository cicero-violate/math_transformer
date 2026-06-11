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
    triton_block_d: int | None = None
    triton_block_k: int | None = None
    effective_triton_block_d: int | None = None
    effective_triton_block_k: int | None = None
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
    topology_prepare_ms: float = 0.0
    learned_scorer_ms: float = 0.0
    neighbor_table_build_ms: float = 0.0
    total_with_prepare_ms: float = 0.0
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
    profile_fused_norm_qkv: bool = False
    profile_fused_attn_outproj: bool = False
    # End-to-end
    full_e2e_ms: float = 0.0
    masked_e2e_ms: float = 0.0
    sparse_e2e_uncached_ms: float = 0.0
    sparse_e2e_cached_ms: float = 0.0
    # Relation diagnostics
    by_relation: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"n={self.n}  mode={self.node_mode}  k={self.k}  device={self.device}  "
            f"triton_block_d={self.triton_block_d}  triton_block_k={self.triton_block_k}  "
            f"effective_triton_block_d={self.effective_triton_block_d}  "
            f"effective_triton_block_k={self.effective_triton_block_k}",
            f"edges: full={self.full_edges}  allowed={self.allowed_edges}",
            f"avg_k={self.avg_k:.2f}  max_k={self.max_k}  padding={self.padding_ratio:.3f}",
            f"sparsity={self.sparsity_ratio:.4f}  rel_reduce={self.relation_reduction:.4f}",
            "--- topology build ---",
            f"  build_ms={self.topology_build_ms:.3f}ms  "
            f"topology_prepare={self.topology_prepare_ms:.3f}ms  "
            f"learned_scorer={self.learned_scorer_ms:.3f}ms  "
            f"neighbor_table_build={self.neighbor_table_build_ms:.3f}ms  "
            f"total_with_prepare={self.total_with_prepare_ms:.3f}ms",
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
    correct_count: int | None = None
    correct_by_example: list[bool] = field(default_factory=list)
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
            f"route_acc={self.route_accuracy:.6f}"
        )
        if self.correct_count is not None:
            line += f"  correct={self.correct_count}/{self.n_examples}"
        line += f"{agree}{hidden}{logits}"
        if not self.by_expert:
            return line
        details = "  ".join(
            f"{expert}={stats['correct']}/{stats['total']}({stats['accuracy']:.6f})"
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
    triton_block_d: int | None = None,
    triton_block_k: int | None = None,
    benchmark_seed: int | None = None,
) -> BenchmarkReport:
    from .model import MathRoutedTransformer, MathRoutedTransformerBlock
    from .topology import TopologyBuilder
    from .learned_topology_runtime import LearnedTopologyBuilder
    from .block_learned_topology import HeuristicBlockTopologyBuilder
    from .embedder import MathEmbedder
    from .learned_topology import FEATURE_NAMES, topk_mask_from_scores
    from .topology import TopologyBuilder
    from .topology_trace import (
        TopologyTraceWriter,
        hash_nodes,
        summarize_mask,
        summarize_overlap,
        summarize_scores,
    )
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

    def _reset_benchmark_rng() -> None:
        if benchmark_seed is None:
            return
        torch.manual_seed(benchmark_seed)
        torch.cuda.manual_seed_all(benchmark_seed)
        np.random.seed(benchmark_seed)

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
    base_topology_mode = "middle_preserving_topk" if topology_mode == "learned_block_topk" else topology_mode

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
            topology_mode=base_topology_mode,
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
    _reset_benchmark_rng()
    B, H, Dh = 1, n_heads, d_model // n_heads
    effective_triton_block_d = triton_block_d if triton_block_d is not None else 1 << (Dh - 1).bit_length()
    effective_triton_block_k = triton_block_k if triton_block_k is not None else 1 << (max(trunc_k, 1) - 1).bit_length()
    q_ = torch.randn(B, H, n, Dh, device=device)
    k_ = torch.randn(B, H, n, Dh, device=device)
    v_ = torch.randn(B, H, n, Dh, device=device)
    mask_4d = mask_t.unsqueeze(0).unsqueeze(0)

    dense_full_attn_ms   = _timed(lambda: math_attention(q_, k_, v_, None), n_warmup, n_iter)
    dense_masked_attn_ms = _timed(lambda: math_attention(q_, k_, v_, mask_4d), n_warmup, n_iter)
    nbr_exact_ms = _timed(lambda: triton_neighbor_attention(q_, k_, v_, nb_exact, valid_exact, block_d=triton_block_d, block_k=triton_block_k), n_warmup, n_iter)
    nbr_trunc_ms = _timed(lambda: triton_neighbor_attention(q_, k_, v_, nb_trunc, valid_trunc, block_d=triton_block_d, block_k=triton_block_k), n_warmup, n_iter)
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
        lambda: triton_neighbor_attention(q_, k_, v_, _nb_stk, _valid_stk, block_d=triton_block_d, block_k=triton_block_k),
        n_warmup, n_iter,
    )

    priority_t = torch.tensor(priority, dtype=torch.int8, device=device)
    symbolic_scores_t = symbolic_priority_scores(priority_t)

    def _attn_for_selector(selector: str) -> torch.Tensor:
        if selector == "topology_only":
            return triton_neighbor_attention(q_, k_, v_, nb_trunc, valid_trunc, block_d=triton_block_d, block_k=triton_block_k)
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
        return triton_neighbor_attention(q_, k_, v_, nb_sel, valid_sel, block_d=triton_block_d, block_k=triton_block_k)

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
    _reset_benchmark_rng()
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
        triton_block_d=triton_block_d,
        triton_block_k=triton_block_k,
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
        triton_block_d=triton_block_d,
        triton_block_k=triton_block_k,
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
            triton_block_d=triton_block_d,
            triton_block_k=triton_block_k,
        ).to(device)
        _install_learned_topology(block_prepared)
        block_prepared.eval()
        with torch.no_grad():
            block_prepared.prepare_static_topology(nodes, env=env or None, device=torch.device(device))
            block_prepared.profile_cached_sparse_block(
                x_dm, nodes, env=env or None,
                fused_norm_qkv=profile_fused_norm_qkv,
                fused_attn_outproj=profile_fused_attn_outproj,
            )

        def _profile_prepared_once() -> dict[str, float]:
            with torch.no_grad():
                _, timings = block_prepared.profile_cached_sparse_block(
                    x_dm, nodes, env=env or None,
                    fused_norm_qkv=profile_fused_norm_qkv,
                    fused_attn_outproj=profile_fused_attn_outproj,
                )
            return timings

        prepared_samples = []
        with torch.no_grad():
            for _ in range(n_warmup):
                _profile_prepared_once()
            for _ in range(n_iter):
                prepared_samples.append(_profile_prepared_once())

        prepared_totals = np.array([t.get("total_block_ms", 0.0) for t in prepared_samples])
        prepared_attns = np.array([
            t.get("attention_outproj_ms", t.get("attention_kernel_ms", 0.0))
            for t in prepared_samples
        ])
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
            triton_block_d=triton_block_d,
            triton_block_k=triton_block_k,
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
              middle_bridge_width=middle_bridge_width,
              triton_block_d=triton_block_d, triton_block_k=triton_block_k)
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
        triton_block_d=triton_block_d,
        triton_block_k=triton_block_k,
        effective_triton_block_d=effective_triton_block_d,
        effective_triton_block_k=effective_triton_block_k,
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
        profile_fused_norm_qkv=profile_fused_norm_qkv,
        profile_fused_attn_outproj=profile_fused_attn_outproj,
        full_e2e_ms=full_e2e_ms,
        masked_e2e_ms=masked_e2e_ms,
        sparse_e2e_uncached_ms=sparse_uc_e2e_ms,
        sparse_e2e_cached_ms=sparse_c_e2e_ms,
        by_relation=diag.by_relation,
    )

    if save_dir:
        _save_report(report, save_dir)

    return report


def run_paired_learned_topology_benchmark(
    n: int,
    node_mode: str = "trees",
    d_model: int = 64,
    n_heads: int = 4,
    d_ff: int = 128,
    topk: int = 3,
    local_window: int = 1,
    hand_k: int = 16,
    learned_k: int = 6,
    n_warmup: int = 3,
    n_iter: int = 100,
    exprs: list[str] | None = None,
    examples_path: str | None = None,
    middle_bridge_width: int = 1,
    learned_scorer_checkpoint: str | None = None,
    hand_triton_block_d: int | None = None,
    hand_triton_block_k: int | None = None,
    learned_triton_block_d: int | None = None,
    learned_triton_block_k: int | None = None,
    benchmark_seed: int | None = None,
    hand_save_dir: str | None = None,
    learned_save_dir: str | None = None,
    profile_fused_norm_qkv: bool = False,
    profile_fused_attn_outproj: bool = False,
    topology_mode: str = "learned_topology",
    block_size: int = 64,
    topk_blocks: int = 4,
    block_local_window: int = 1,
    block_token_cap: int = 16,
    native_block_sparse_attn: bool = False,
    protect_noncommutative: bool = False,
    polarity_summary: str | None = None,
    polarity_alpha: float = 0.0,
) -> tuple[BenchmarkReport, BenchmarkReport]:
    """Benchmark hand and learned prepared topologies with one shared block/input.

    This isolates topology/neighbor-table runtime effects from K-independent block
    variance by using identical LayerNorm/QKV/out-proj/FFN weights and identical
    input for both topology timings.
    """
    from .model import MathRoutedTransformerBlock
    from .topology import TopologyBuilder
    from .learned_topology_runtime import LearnedTopologyBuilder
    from .block_learned_topology import HeuristicBlockTopologyBuilder
    from .topology_cache import TopologyCache, PreparedTopology
    from .embedder import MathEmbedder
    from .triton_attention import TRITON_AVAILABLE
    import torch.nn as nn

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: CPU benchmark path has been removed")
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for neighbor-sparse attention")
    if topology_mode == "learned_topology" and not learned_scorer_checkpoint:
        raise ValueError("--learned-scorer-checkpoint is required for paired learned benchmark")
    device = torch.device("cuda")

    def _reset_benchmark_rng() -> None:
        if benchmark_seed is None:
            return
        torch.manual_seed(benchmark_seed)
        torch.cuda.manual_seed_all(benchmark_seed)
        np.random.seed(benchmark_seed)

    def _next_pow2(x: int) -> int:
        return 1 << (max(int(x), 1) - 1).bit_length()

    def _effective_block_d(block_d: int | None, d_head: int) -> int:
        return int(block_d) if block_d is not None else _next_pow2(d_head)

    def _effective_block_k(block_k: int | None, k: int) -> int:
        return int(block_k) if block_k is not None else _next_pow2(k)

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
    nodes = _collect_nodes_trees(exprs, n) if node_mode == "trees" else _collect_nodes_roots(exprs, n)
    embedder = MathEmbedder()
    z = embedder.encode_batch(nodes)
    z_t = torch.from_numpy(z).float()

    hand_builder = TopologyBuilder(
        topk=topk,
        local_window=local_window,
        topology_mode="middle_preserving_topk",
        fixed_k=hand_k,
        middle_bridge_width=middle_bridge_width,
    )
    if topology_mode == "learned_block_topk":
        learned_builder = HeuristicBlockTopologyBuilder(
            block_size=block_size,
            topk_blocks=topk_blocks,
            include_local_blocks=block_local_window,
            block_token_cap=block_token_cap,
            fixed_k=learned_k,
            topk=topk,
            local_window=local_window,
            middle_bridge_width=middle_bridge_width,
            device=device,
            prepare_mode="native_block_only" if native_block_sparse_attn else "full",
        )
    else:
        learned_builder = LearnedTopologyBuilder(
            learned_scorer_checkpoint or "",
            fixed_k=learned_k,
            topk=topk,
            local_window=local_window,
            middle_bridge_width=middle_bridge_width,
            device=device,
            protect_noncommutative=protect_noncommutative,
            polarity_summary=polarity_summary,
            polarity_alpha=polarity_alpha,
        )
    cache = TopologyCache()

    def _prepare_with_timing(builder, k: int):
        _sync()
        t0 = time.perf_counter()
        prepared = cache.get_or_prepare(nodes, z, env or None, builder, max_neighbors=k, device=device)
        _sync()
        total = (time.perf_counter() - t0) * 1000.0
        timing = dict(getattr(builder, "last_timing", {}) or {})
        timing.setdefault("topology_prepare_ms", total)
        timing.setdefault("learned_scorer_ms", total if getattr(builder, "is_learned_topology", False) else 0.0)
        timing.setdefault("neighbor_table_build_ms", 0.0)
        return prepared, timing

    hand_prepared, hand_prepare_timing = _prepare_with_timing(hand_builder, hand_k)
    learned_prepared, learned_prepare_timing = _prepare_with_timing(learned_builder, learned_k)

    _reset_benchmark_rng()
    proj = nn.Linear(z_t.shape[1], d_model, bias=False).to(device)
    with torch.no_grad():
        x_dm = proj(z_t.to(device).unsqueeze(0))

    hand_block = MathRoutedTransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        topk=topk,
        local_window=local_window,
        attention_mode="neighbor_sparse",
        max_neighbors=hand_k,
        topology_cache=TopologyCache(),
        topology_mode="middle_preserving_topk",
        fixed_k=hand_k,
        middle_bridge_width=middle_bridge_width,
        triton_block_d=hand_triton_block_d,
        triton_block_k=hand_triton_block_k,
    ).to(device)
    hand_block.eval()

    learned_attention_mode = "block_sparse" if bool(native_block_sparse_attn and learned_prepared.is_block_topology) else "neighbor_sparse"
    learned_block = MathRoutedTransformerBlock(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        topk=topk,
        local_window=local_window,
        attention_mode=learned_attention_mode,
        max_neighbors=learned_k,
        topology_cache=TopologyCache(),
        topology_mode="learned_block_topk" if learned_prepared.is_block_topology else topology_mode,
        fixed_k=learned_k,
        middle_bridge_width=middle_bridge_width,
        triton_block_d=learned_triton_block_d,
        triton_block_k=learned_triton_block_k,
    ).to(device)
    learned_block.eval()
    learned_block.load_state_dict(hand_block.state_dict(), strict=False)

    # The paired benchmark is intended to isolate topology/attention effects.
    # Share the actual non-topology modules so QKV/out-proj, LayerNorm, FFN,
    # dropout, and parameter-cache pointer differences do not dominate the
    # final few microseconds of the strict comparison.
    learned_block.norm1 = hand_block.norm1
    learned_block.norm2 = hand_block.norm2
    learned_block.ff = hand_block.ff
    learned_block.drop = hand_block.drop
    learned_block.attn.q_proj = hand_block.attn.q_proj
    learned_block.attn.k_proj = hand_block.attn.k_proj
    learned_block.attn.v_proj = hand_block.attn.v_proj
    learned_block.attn.out_proj = hand_block.attn.out_proj

    if hasattr(hand_block.attn, "_fused_qkv_weight") and hasattr(learned_block.attn, "_fused_qkv_weight"):
        shared_qkv_weight = hand_block.attn._fused_qkv_weight()
        if hasattr(learned_block.attn, "_qkv_weight_cache"):
            learned_block.attn._qkv_weight_cache = shared_qkv_weight
        if hasattr(learned_block.attn, "_qkv_weight_versions"):
            learned_block.attn._qkv_weight_versions = (
                learned_block.attn.q_proj.weight._version,
                learned_block.attn.k_proj.weight._version,
                learned_block.attn.v_proj.weight._version,
            )

    def _install_prepared_static(
        block_obj: MathRoutedTransformerBlock,
        prepared: PreparedTopology,
        triton_block_d: int | None,
        triton_block_k: int | None,
    ) -> None:
        block_obj.max_neighbors = learned_k if prepared.neighbors is None else prepared.k
        block_obj._prepared_topology = prepared
        block_obj.static_neighbors = (
            torch.empty(0, 0, dtype=torch.long, device=device)
            if prepared.neighbors is None else prepared.neighbors.to(device).contiguous()
        )
        block_obj.static_valid_i8 = (
            torch.empty(0, 0, dtype=torch.int8, device=device)
            if prepared.valid_i8 is None else prepared.valid_i8.to(device).contiguous()
        )
        block_obj.static_block_neighbors = (
            torch.empty(0, 0, dtype=torch.long, device=device)
            if prepared.block_neighbors is None else prepared.block_neighbors.to(device).contiguous()
        )
        block_obj.static_block_valid_i8 = (
            torch.empty(0, 0, dtype=torch.int8, device=device)
            if prepared.block_valid_i8 is None else prepared.block_valid_i8.to(device).contiguous()
        )
        block_obj.static_block_token_indices = (
            torch.empty(0, 0, 0, dtype=torch.long, device=device)
            if prepared.block_token_indices is None else prepared.block_token_indices.to(device).contiguous()
        )
        block_obj.static_block_token_valid_i8 = (
            torch.empty(0, 0, 0, dtype=torch.int8, device=device)
            if prepared.block_token_valid_i8 is None else prepared.block_token_valid_i8.to(device).contiguous()
        )
        block_obj.static_block_size = prepared.block_size
        if hasattr(block_obj.attn, "triton_block_d"):
            block_obj.attn.triton_block_d = triton_block_d
            block_obj.attn.triton_block_k = triton_block_k

    _install_prepared_static(hand_block, hand_prepared, hand_triton_block_d, hand_triton_block_k)
    _install_prepared_static(learned_block, learned_prepared, learned_triton_block_d, learned_triton_block_k)

    def _profile_once(block_obj: MathRoutedTransformerBlock) -> dict[str, float]:
        with torch.no_grad():
            _, timings = block_obj.profile_cached_sparse_block(
                x_dm, nodes, env=env or None,
                fused_norm_qkv=profile_fused_norm_qkv,
                fused_attn_outproj=profile_fused_attn_outproj,
            )
        return timings

    hand_samples: list[dict[str, float]] = []
    learned_samples: list[dict[str, float]] = []
    with torch.no_grad():
        for _ in range(n_warmup):
            _profile_once(hand_block)
            _profile_once(learned_block)
        for _ in range(n_iter):
            hand_samples.append(_profile_once(hand_block))
            learned_samples.append(_profile_once(learned_block))

    bucket_keys = [
        "topology_prepare_ms",
        "norm1_ms",
        "qkv_ms",
        "norm_qkv_ms",
        "attention_kernel_ms",
        "out_proj_ms",
        "attention_outproj_ms",
        "native_block_backend_triton",
        "native_block_backend_vectorized",
        "residual1_ms",
        "norm2_ms",
        "ffn_ms",
        "residual2_ms",
        "total_block_ms",
    ]

    def _median_buckets(samples: list[dict[str, float]]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key in bucket_keys:
            vals = [float(s.get(key, 0.0)) for s in samples]
            out[key] = float(np.median(np.array(vals))) if vals else 0.0
        return out

    hand_buckets = _median_buckets(hand_samples)
    learned_buckets = _median_buckets(learned_samples)

    def _make_report(
        prepared: PreparedTopology,
        buckets: dict[str, float],
        triton_block_d: int | None,
        triton_block_k: int | None,
        prepare_timing: dict[str, float],
    ) -> BenchmarkReport:
        diag = prepared.diagnostics
        total = float(buckets.get("total_block_ms", 0.0))
        attn_outproj = float(buckets.get("attention_outproj_ms", 0.0) or 0.0)
        attn_kernel = float(buckets.get("attention_kernel_ms", 0.0) or 0.0)
        attn = attn_outproj if attn_outproj > 0.0 else attn_kernel
        non_attn = max(0.0, total - attn)
        topology_prepare_ms = float(prepare_timing.get("topology_prepare_ms", 0.0))
        learned_scorer_ms = float(prepare_timing.get("learned_scorer_ms", 0.0))
        neighbor_table_build_ms = float(prepare_timing.get("neighbor_table_build_ms", 0.0))
        total_with_prepare_ms = topology_prepare_ms + total
        selector_results = {
            "paired_prepared_shared_block": {
                **{k: float(v) for k, v in buckets.items()},
                "topology_prepare_ms": topology_prepare_ms,
                "learned_scorer_ms": learned_scorer_ms,
                "neighbor_table_build_ms": neighbor_table_build_ms,
                "prepared_block_ms": total,
                "prepared_attention_ms": attn,
                "prepared_non_attention_ms": non_attn,
                "total_with_prepare_ms": total_with_prepare_ms,
                "same_block_weights": 1.0,
                "same_input": 1.0,
                "profile_fused_norm_qkv": float(profile_fused_norm_qkv),
                "profile_fused_attn_outproj": float(profile_fused_attn_outproj),
                "native_block_sparse_attn": float(native_block_sparse_attn and prepared.is_block_topology),
                "block_count": float(prepared.block_neighbors.shape[0]) if prepared.block_neighbors is not None else 0.0,
                "block_size": float(prepared.block_size or 0),
                "topk_blocks": float(prepared.block_neighbors.shape[1]) if prepared.block_neighbors is not None else 0.0,
                "block_token_cap": float(prepared.block_token_indices.shape[2]) if prepared.block_token_indices is not None else 0.0,
                "native_effective_token_k": float(prepared.block_token_indices.shape[1] * prepared.block_token_indices.shape[2]) if prepared.block_token_indices is not None else 0.0,
            }
        }
        return BenchmarkReport(
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
            device=str(device),
            triton_block_d=triton_block_d,
            triton_block_k=triton_block_k,
            effective_triton_block_d=_effective_block_d(triton_block_d, d_model // n_heads),
            effective_triton_block_k=_effective_block_k(triton_block_k, learned_k if prepared.k == 0 else prepared.k),
            triton_sparse_attn_ms=attn,
            sparse_block_cached_ms=total,
            prepared_static_sparse_block_ms=total,
            prepared_static_sparse_attention_ms=attn,
            prepared_static_sparse_non_attention_ms=non_attn,
            topology_prepare_ms=topology_prepare_ms,
            learned_scorer_ms=learned_scorer_ms,
            neighbor_table_build_ms=neighbor_table_build_ms,
            total_with_prepare_ms=total_with_prepare_ms,
            profile_fused_norm_qkv=profile_fused_norm_qkv,
            profile_fused_attn_outproj=profile_fused_attn_outproj,
            selector_results=selector_results,
            by_relation=diag.by_relation,
        )

    hand_report = _make_report(hand_prepared, hand_buckets, hand_triton_block_d, hand_triton_block_k, hand_prepare_timing)
    learned_report = _make_report(learned_prepared, learned_buckets, learned_triton_block_d, learned_triton_block_k, learned_prepare_timing)

    if hand_save_dir:
        _save_report(hand_report, hand_save_dir)
    if learned_save_dir:
        _save_report(learned_report, learned_save_dir)
    return hand_report, learned_report


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
    protect_noncommutative: bool = False,
    polarity_summary: str | None = None,
    polarity_alpha: float = 0.0,
    block_size: int = 64,
    topk_blocks: int = 4,
    block_local_window: int = 1,
    block_token_cap: int = 16,
    trace_output: str | None = None,
) -> list[QualityReport]:
    from .model import MathRoutedTransformer
    from .parser import parse
    from .normalize import normalize
    from .learned_topology_runtime import LearnedTopologyBuilder
    from .block_learned_topology import HeuristicBlockTopologyBuilder
    from .embedder import MathEmbedder
    from .learned_topology import FEATURE_NAMES, topk_mask_from_scores
    from .topology import TopologyBuilder
    from .topology_trace import (
        TopologyTraceWriter,
        hash_nodes,
        summarize_mask,
        summarize_overlap,
        summarize_scores,
    )

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
    base_topology_mode = "middle_preserving_topk" if topology_mode == "learned_block_topk" else topology_mode
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
            correct_count=sum(1 for pred, target in zip(dense_preds, targets) if pred == target),
            correct_by_example=[pred == target for pred, target in zip(dense_preds, targets)],
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
                correct_count=sum(1 for pred, target in zip(sparse_preds, targets) if pred == target),
                correct_by_example=[pred == target for pred, target in zip(sparse_preds, targets)],
                dense_agreement=op_accuracy(sparse_preds, dense_preds),
                hidden_l1=agreement["hidden_l1"],
                hidden_cos=agreement["hidden_cos"],
                logit_l1=agreement["logit_l1"],
                logit_kl_dense_to_sparse=agreement["logit_kl_dense_to_sparse"],
                by_expert=_by_expert(sparse_preds),
            )
        )

    if topology_mode == "learned_block_topk":
        block_model = MathRoutedTransformer(
            **base_kwargs,
            attention_mode="neighbor_sparse",
            max_neighbors=learned_k,
            share_topology_cache=True,
            sparse_selector="topology_only",
        ).to(dev)
        block_model.load_state_dict(dense_state)
        for layer in block_model.layers:
            layer.topology = HeuristicBlockTopologyBuilder(
                block_size=block_size,
                topk_blocks=topk_blocks,
                include_local_blocks=block_local_window,
                block_token_cap=block_token_cap,
                fixed_k=learned_k,
                topk=topk,
                local_window=local_window,
                middle_bridge_width=middle_bridge_width,
                device=dev,
            )
            layer.max_neighbors = learned_k
        block_preds, block_hiddens, block_logits = _forward_records(block_model, pass_nodes=True)
        agreement = _agreement_metrics(
            sparse_hiddens=block_hiddens,
            sparse_logits=block_logits,
            dense_hiddens=dense_hiddens,
            dense_logits=dense_logits,
        )
        reports.append(
            QualityReport(
                mode="learned_block_topk",
                k=learned_k,
                n_examples=len(records),
                route_accuracy=op_accuracy(block_preds, targets),
                correct_count=sum(1 for pred, target in zip(block_preds, targets) if pred == target),
                correct_by_example=[pred == target for pred, target in zip(block_preds, targets)],
                dense_agreement=op_accuracy(block_preds, dense_preds),
                hidden_l1=agreement["hidden_l1"],
                hidden_cos=agreement["hidden_cos"],
                logit_l1=agreement["logit_l1"],
                logit_kl_dense_to_sparse=agreement["logit_kl_dense_to_sparse"],
                by_expert=_by_expert(block_preds),
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
                protect_noncommutative=protect_noncommutative,
                polarity_summary=polarity_summary,
                polarity_alpha=polarity_alpha,
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
                correct_count=sum(1 for pred, target in zip(learned_preds, targets) if pred == target),
                correct_by_example=[pred == target for pred, target in zip(learned_preds, targets)],
                dense_agreement=op_accuracy(learned_preds, dense_preds),
                hidden_l1=agreement["hidden_l1"],
                hidden_cos=agreement["hidden_cos"],
                logit_l1=agreement["logit_l1"],
                logit_kl_dense_to_sparse=agreement["logit_kl_dense_to_sparse"],
                by_expert=_by_expert(learned_preds),
            )
        )


    if trace_output:
        trace_builder = None
        if learned_scorer_checkpoint:
            trace_builder = LearnedTopologyBuilder(
                learned_scorer_checkpoint,
                fixed_k=learned_k,
                topk=topk,
                local_window=local_window,
                middle_bridge_width=middle_bridge_width,
                device=dev,
                protect_noncommutative=protect_noncommutative,
                polarity_summary=polarity_summary,
                polarity_alpha=polarity_alpha,
            )
        hand_builder = TopologyBuilder(
            topk=topk,
            local_window=local_window,
            topology_mode="middle_preserving_topk" if topology_mode == "learned_block_topk" else topology_mode,
            fixed_k=fixed_k,
            middle_bridge_width=middle_bridge_width,
            relation_weights=relation_weights,
        )
        embedder = MathEmbedder()
        with TopologyTraceWriter(trace_output) as writer:
            for sample_idx, rec in enumerate(records):
                root = normalize(parse(rec["expr"]))
                nodes = root.collect_nodes()
                env = rec["env"] or None
                z = embedder.encode_batch(nodes)
                hand_mask_np, _ = hand_builder.build_scored_topk(nodes, z, env)
                hand_mask = torch.as_tensor(hand_mask_np, dtype=torch.bool)
                learned_mask = None
                learned_scores = None
                if trace_builder is not None:
                    learned_scores = trace_builder._scores(nodes, z, env, dev)  # compact trace path only
                    learned_mask = topk_mask_from_scores(learned_scores, learned_k)
                prediction: dict[str, object] = {
                    "target_expert": rec.get("expert"),
                    "target_expert_id": rec.get("expert_id"),
                    "dense_pred_id": dense_preds[sample_idx],
                    "dense_correct": dense_preds[sample_idx] == targets[sample_idx],
                }
                if k_values:
                    # The first topology_only row is the normal hand/sparse comparison used by quality eval.
                    # Additional k rows remain available in aggregate reports.
                    pass
                if learned_scorer_checkpoint:
                    prediction.update({
                        "learned_pred_id": learned_preds[sample_idx],
                        "learned_correct": learned_preds[sample_idx] == targets[sample_idx],
                        "learned_dense_agree": learned_preds[sample_idx] == dense_preds[sample_idx],
                    })
                    hidden_l1 = float((learned_hiddens[sample_idx] - dense_hiddens[sample_idx]).abs().mean().item())
                    hidden_cos = float(torch.nn.functional.cosine_similarity(
                        learned_hiddens[sample_idx].reshape(1, -1),
                        dense_hiddens[sample_idx].reshape(1, -1),
                        dim=1,
                    ).item())
                    logit_l1 = float((learned_logits[sample_idx] - dense_logits[sample_idx]).abs().mean().item())
                    logit_kl = float(torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(learned_logits[sample_idx], dim=-1),
                        torch.nn.functional.softmax(dense_logits[sample_idx], dim=-1),
                        reduction="batchmean",
                    ).item())
                else:
                    hidden_l1 = hidden_cos = logit_l1 = logit_kl = None
                writer.write({
                    "sample_id": sample_idx,
                    "domain": "math",
                    "expr": rec.get("expr"),
                    "nodes_hash": hash_nodes(nodes),
                    "n": len(nodes),
                    "k": learned_k if learned_scorer_checkpoint else fixed_k,
                    "scorer_checkpoint": learned_scorer_checkpoint,
                    "feature_schema": "topology_edge_features.v1",
                    "feature_names": list(FEATURE_NAMES),
                    "topology_config": {
                        "topology_mode": topology_mode,
                        "fixed_k": fixed_k,
                        "learned_k": learned_k,
                        "topk": topk,
                        "local_window": local_window,
                        "middle_bridge_width": middle_bridge_width,
                        "n_layers": n_layers,
                    },
                    "scores": summarize_scores(learned_scores) if learned_scores is not None else {},
                    "target_topology": summarize_mask(hand_mask),
                    "pred_topology": summarize_mask(learned_mask) if learned_mask is not None else {},
                    "overlap": summarize_overlap(learned_mask, hand_mask) if learned_mask is not None else {},
                    "prediction": prediction,
                    "agreement": {
                        "hidden_l1": hidden_l1,
                        "hidden_cos": hidden_cos,
                        "logit_l1": logit_l1,
                        "logit_kl": logit_kl,
                    },
                    "diagnostics": {
                        "trace_source": "run_quality_eval",
                        "has_learned_topology": bool(learned_scorer_checkpoint),
                    },
                })

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
        f"{'n':>5}  {'mode':>5}  {'tb_d':>5}  {'tb_k':>5}  {'eff_k':>5}  {'allowed':>8}  {'full':>6}  "
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
        f"{r.n:>5}  {r.node_mode:>5}  {str(r.triton_block_d):>5}  {str(r.triton_block_k):>5}  {str(r.effective_triton_block_k):>5}  {r.allowed_edges:>8}  {r.full_edges:>6}  "
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


def _run_paired_learned_topology_benchmark_cli(args: argparse.Namespace) -> None:
    sizes = [int(x) for x in args.sizes.split(",")]
    modes = args.node_mode.split(",")
    if len(sizes) != 1 or len(modes) != 1:
        raise ValueError("paired learned benchmark expects exactly one size and one node mode")
    hand_report, learned_report = run_paired_learned_topology_benchmark(
        n=sizes[0],
        node_mode=modes[0],
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        topk=args.topk,
        local_window=args.local_window,
        hand_k=args.fixed_k,
        learned_k=args.learned_k,
        n_warmup=args.warmup,
        n_iter=args.iters,
        examples_path=args.examples,
        middle_bridge_width=args.middle_bridge_width,
        learned_scorer_checkpoint=args.learned_scorer_checkpoint,
        hand_triton_block_d=args.hand_triton_block_d,
        hand_triton_block_k=args.hand_triton_block_k,
        learned_triton_block_d=args.learned_triton_block_d,
        learned_triton_block_k=args.learned_triton_block_k,
        benchmark_seed=args.benchmark_seed,
        hand_save_dir=args.hand_save_dir,
        learned_save_dir=args.learned_save_dir,
        profile_fused_norm_qkv=args.profile_fused_norm_qkv,
        profile_fused_attn_outproj=args.profile_fused_attn_outproj,
        topology_mode=args.topology_mode,
        block_size=args.block_size,
        topk_blocks=args.topk_blocks,
        block_local_window=args.block_local_window,
        block_token_cap=args.block_token_cap,
        native_block_sparse_attn=args.native_block_sparse_attn,
        protect_noncommutative=args.protect_noncommutative,
        polarity_summary=args.polarity_summary,
        polarity_alpha=args.polarity_alpha,
    )
    print("paired_prepared_shared_block=true same_input=true same_block_weights=true")
    print(f"fused_norm_qkv={args.profile_fused_norm_qkv} fused_attn_outproj={args.profile_fused_attn_outproj} native_block_sparse_attn={args.native_block_sparse_attn}")
    print("hand")
    print(hand_report)
    print("learned")
    print(learned_report)


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
                profile_fused_norm_qkv=args.profile_fused_norm_qkv,
                profile_fused_attn_outproj=args.profile_fused_attn_outproj,
                learned_scorer_checkpoint=args.learned_scorer_checkpoint,
                learned_k=args.learned_k,
                triton_block_d=args.triton_block_d,
                triton_block_k=args.triton_block_k,
                benchmark_seed=args.benchmark_seed,
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
        protect_noncommutative=args.protect_noncommutative,
        polarity_summary=args.polarity_summary,
        polarity_alpha=args.polarity_alpha,
        block_size=args.block_size,
        topk_blocks=args.topk_blocks,
        block_local_window=args.block_local_window,
        block_token_cap=args.block_token_cap,
        trace_output=args.trace_output,
    )
    print(f"examples={args.examples}  checkpoint={args.checkpoint or 'random_init'}")
    for report in reports:
        print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval / benchmark Math-Routed Transformer")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--paired-learned-topology-benchmark", action="store_true", dest="paired_learned_topology_benchmark")
    parser.add_argument('--profile-prepared-block', action="store_true", dest="profile_prepared_block")
    parser.add_argument("--profile-fused-norm-qkv", action="store_true", dest="profile_fused_norm_qkv")
    parser.add_argument("--profile-fused-attn-outproj", action="store_true", dest="profile_fused_attn_outproj")
    parser.add_argument("--native-block-sparse-attn", action="store_true", dest="native_block_sparse_attn")
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
                        choices=["union", "scored_topk", "middle_preserving_topk", "learned_topology", "learned_block_topk"])
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
    parser.add_argument("--protect-noncommutative", action="store_true", dest="protect_noncommutative")
    parser.add_argument("--polarity-summary", default=None, dest="polarity_summary")
    parser.add_argument("--polarity-alpha", type=float, default=0.0, dest="polarity_alpha")
    parser.add_argument("--block-size", type=int, default=64, dest="block_size")
    parser.add_argument("--topk-blocks", type=int, default=4, dest="topk_blocks")
    parser.add_argument("--block-local-window", type=int, default=1, dest="block_local_window")
    parser.add_argument("--block-token-cap", type=int, default=16, dest="block_token_cap")
    parser.add_argument("--triton-block-d", type=int, default=None, dest="triton_block_d")
    parser.add_argument("--triton-block-k", type=int, default=None, dest="triton_block_k")
    parser.add_argument("--benchmark-seed", type=int, default=None, dest="benchmark_seed")
    parser.add_argument("--hand-triton-block-d", type=int, default=None, dest="hand_triton_block_d")
    parser.add_argument("--hand-triton-block-k", type=int, default=None, dest="hand_triton_block_k")
    parser.add_argument("--learned-triton-block-d", type=int, default=None, dest="learned_triton_block_d")
    parser.add_argument("--learned-triton-block-k", type=int, default=None, dest="learned_triton_block_k")
    parser.add_argument("--hand-save-dir", default=None, dest="hand_save_dir")
    parser.add_argument("--learned-save-dir", default=None, dest="learned_save_dir")
    parser.add_argument("--save-dir", default=None, dest="save_dir")
    parser.add_argument("--trace-output", default=None, dest="trace_output", help="Optional JSONL path for compact quality/topology traces.")
    args = parser.parse_args()
    if args.paired_learned_topology_benchmark:
        _run_paired_learned_topology_benchmark_cli(args)
    if args.benchmark:
        _run_benchmark_cli(args)
    if args.quality:
        _run_quality_cli(args)


if __name__ == "__main__":
    main()
