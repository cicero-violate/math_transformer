"""
Recurrent Frontier Graph Transformer — outer loop and benchmark (P1.8).

Forward loop (plan formula):
  H_t = F_θ^L(X_t, G_t)
  C_t = BoundaryCandidates(∂G_t, G_world)
  G_{t+1} = KeepTopB(G_t ∪ TopK(Score(H_t, C_t)))
  G_world^{t+1} = Accept(G_world^t ⊕ ΔG_t)

Effective reach:    D_effective ≈ L * T_outer
Active compute:     O(T_outer * L * |G_t| * K * d)

First valid v21 experiment (plan §First Valid v21 Experiment):
  K=4/8, L=2, T_outer ∈ {1,2,3}, H ∈ {1,2,3,4}, |G_t| ≤ B
  Acceptance: J(T_outer=3, closure) > J(T_outer=1, one_hop_topk)

Non-negotiable gate 7: recurrent frontier claims must compare T_outer=1
against T_outer>1 under the same K, L, B, and evaluator.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .world_graph import WorldGraph, ActiveGraph
from .frontier_planner import (
    FrontierExpansionResult,
    OneHopTopKPlanner,
    ClosureAwarePlanner,
)
from .graph_closure import ClosureMode

SCHEMA_VERSION = "recurrent_frontier.v1"


@dataclass
class RecurrentFrontierConfig:
    """
    Configuration for one recurrent frontier run.

    k:              sparse fanout K per attention layer
    l:              local transformer depth L
    t_outer_values: which T_outer values to sweep
    budget:         active graph node budget B
    h_horizon:      Kleene closure horizon H
    planner:        "one_hop_topk" | "closure_boolean" | "closure_cost" | "closure_utility"
    device:         "cpu" | "cuda"
    seed:           RNG seed for reproducibility
    """
    k: int = 4
    l: int = 2
    t_outer_values: list[int] = field(default_factory=lambda: [1, 2, 3])
    budget: int = 32
    h_horizon: int = 2
    planner: str = "one_hop_topk"
    device: str = "cpu"
    seed: int = 42

    def as_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["t_outer_values"] = list(self.t_outer_values)
        return d


@dataclass
class RecurrentFrontierTrace:
    """Full trace of running the recurrent loop for a fixed T_outer."""
    config: RecurrentFrontierConfig
    t_outer: int
    steps: list[FrontierExpansionResult]
    final_active_node_count: int
    final_active_edge_count: int
    total_frontier_expansions: int
    total_closure_path_count: int
    total_closure_compute_cost: int
    wall_ms: float
    model_fn_error_count: int = 0
    last_model_fn_error: str = ""
    quality_metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.as_dict(),
            "t_outer": self.t_outer,
            "steps": [s.as_dict() for s in self.steps],
            "final_active_node_count": self.final_active_node_count,
            "final_active_edge_count": self.final_active_edge_count,
            "total_frontier_expansions": self.total_frontier_expansions,
            "total_closure_path_count": self.total_closure_path_count,
            "total_closure_compute_cost": self.total_closure_compute_cost,
            "wall_ms": self.wall_ms,
            "model_fn_error_count": self.model_fn_error_count,
            "last_model_fn_error": self.last_model_fn_error,
            "quality_metrics": self.quality_metrics,
        }


@dataclass
class RecurrentFrontierComparison:
    """
    Side-by-side comparison of one-hop vs closure planner across T_outer values.
    Produced by run_recurrent_frontier_comparison().
    """
    schema_version: str = SCHEMA_VERSION
    config: RecurrentFrontierConfig = field(default_factory=RecurrentFrontierConfig)
    onehop_traces: list[RecurrentFrontierTrace] = field(default_factory=list)
    closure_traces: list[RecurrentFrontierTrace] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def acceptance_met(self) -> bool | None:
        return self.summary.get("acceptance_met")

    def save_json(self, path: Path) -> None:
        data = {
            "schema_version": self.schema_version,
            "config": self.config.as_dict(),
            "onehop_traces": [t.as_dict() for t in self.onehop_traces],
            "closure_traces": [t.as_dict() for t in self.closure_traces],
            "summary": self.summary,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def format_table(self) -> str:
        rows = self.summary.get("rows", [])
        header = (
            "T_out | OH_nodes | OH_edges | OH_ms   | CL_nodes | CL_edges | "
            "CL_paths | CL_ms   | CL_cost"
        )
        sep = "-" * len(header)
        lines = [header, sep]
        for r in rows:
            lines.append(
                f"{r['t_outer']:<5} | {r['onehop_active_nodes']:<8} | "
                f"{r['onehop_active_edges']:<8} | {r['onehop_wall_ms']:<7.1f} | "
                f"{r['closure_active_nodes']:<8} | {r['closure_active_edges']:<8} | "
                f"{r['closure_path_count']:<8} | {r['closure_wall_ms']:<7.1f} | "
                f"{r['closure_compute_cost']}"
            )
        accept = self.summary.get("acceptance_met")
        lines.append("")
        lines.append(f"Acceptance (J(T=max,closure) > J(T=min,onehop)): {accept}")
        return "\n".join(lines)


# ── Planner factory ───────────────────────────────────────────────────────────

def _make_planner(config: RecurrentFrontierConfig):
    if config.planner == "one_hop_topk":
        return OneHopTopKPlanner(k=config.k)
    mode_map = {
        "closure_boolean": "boolean",
        "closure_cost": "cost",
        "closure_utility": "utility",
    }
    mode: ClosureMode = mode_map.get(config.planner, "utility")  # type: ignore[assignment]
    return ClosureAwarePlanner(k=config.k, h_horizon=config.h_horizon, closure_mode=mode)


# ── J-score (plan §First Valid v21 Experiment) ────────────────────────────────

def j_score(
    trace: RecurrentFrontierTrace,
    lambda_time: float = 0.01,
    gamma_mem: float = 0.001,
    rho_frontier: float = 0.1,
    kappa_closure: float = 1e-6,
    quality: float = 0.0,
) -> float:
    """
    J = Q - λT - γM - ρC_frontier - κC_closure

    quality MUST come from an actual evaluator (route_acc, task score, etc.).
    The default quality=0 makes the comparison cost-only; it will bias toward
    the cheaper planner rather than the more accurate one.  Always pass a real
    quality value from the evaluator before using J to gate promotion decisions.
    """
    return (
        quality
        - lambda_time * trace.wall_ms
        - gamma_mem * trace.final_active_node_count
        - rho_frontier * trace.total_frontier_expansions
        - kappa_closure * trace.total_closure_compute_cost
    )


# ── Core loop ─────────────────────────────────────────────────────────────────

def run_recurrent_frontier(
    world: WorldGraph,
    seed_ids: list[str],
    config: RecurrentFrontierConfig,
    t_outer: int,
    model_fn: Callable[[ActiveGraph, WorldGraph], np.ndarray] | None = None,
) -> RecurrentFrontierTrace:
    """
    Run the recurrent frontier expansion loop for exactly t_outer steps.

    model_fn: optional (active_graph, world) -> np.ndarray of shape (|G_t|, d)
              Represents F_θ^L(X_t, G_t). If None, h_t is None and candidates
              are scored by graph structure features only.

    Returns a RecurrentFrontierTrace with per-step metrics.
    """
    planner = _make_planner(config)
    active = ActiveGraph.seed(seed_ids, config.budget)

    steps: list[FrontierExpansionResult] = []
    t0 = time.perf_counter()
    model_fn_error_count = 0
    last_model_fn_error = ""

    for t in range(t_outer):
        h_t: np.ndarray | None = None
        if model_fn is not None:
            try:
                h_t = model_fn(active, world)
            except Exception as exc:
                model_fn_error_count += 1
                last_model_fn_error = f"step {t}: {type(exc).__name__}: {exc}"
                # h_t stays None; planner proceeds with graph-structure scoring only

        active, step = planner.expand(active, world, h_t, t)
        steps.append(step)

    wall_ms = (time.perf_counter() - t0) * 1000.0
    final_edges = active.active_edges(world)

    return RecurrentFrontierTrace(
        config=config,
        t_outer=t_outer,
        steps=steps,
        final_active_node_count=len(active.node_ids),
        final_active_edge_count=len(final_edges),
        total_frontier_expansions=sum(s.frontier_node_count for s in steps),
        total_closure_path_count=sum(s.closure_path_count for s in steps),
        total_closure_compute_cost=sum(s.closure_compute_cost for s in steps),
        wall_ms=wall_ms,
        model_fn_error_count=model_fn_error_count,
        last_model_fn_error=last_model_fn_error,
    )


def run_recurrent_frontier_comparison(
    world: WorldGraph,
    seed_ids: list[str],
    config: RecurrentFrontierConfig,
    model_fn: Callable[[ActiveGraph, WorldGraph], np.ndarray] | None = None,
) -> RecurrentFrontierComparison:
    """
    Run the first valid v21 experiment.

    For each T in config.t_outer_values, runs both:
      - OneHopTopKPlanner (baseline)
      - ClosureAwarePlanner at configured h_horizon and mode

    Then builds summary + acceptance check.

    Gate 7: T_outer=1 baseline is always included in t_outer_values.
    Gate 15: closure planner results compared against one-hop under same K, H.
    """
    onehop_cfg = RecurrentFrontierConfig(
        k=config.k, l=config.l,
        t_outer_values=config.t_outer_values,
        budget=config.budget, h_horizon=config.h_horizon,
        planner="one_hop_topk",
        device=config.device, seed=config.seed,
    )
    closure_planner = config.planner if config.planner.startswith("closure") else "closure_utility"
    closure_cfg = RecurrentFrontierConfig(
        k=config.k, l=config.l,
        t_outer_values=config.t_outer_values,
        budget=config.budget, h_horizon=config.h_horizon,
        planner=closure_planner,
        device=config.device, seed=config.seed,
    )

    onehop_traces = [
        run_recurrent_frontier(world, seed_ids, onehop_cfg, t, model_fn)
        for t in config.t_outer_values
    ]
    closure_traces = [
        run_recurrent_frontier(world, seed_ids, closure_cfg, t, model_fn)
        for t in config.t_outer_values
    ]

    summary = _build_summary(onehop_traces, closure_traces, config)
    return RecurrentFrontierComparison(
        config=config,
        onehop_traces=onehop_traces,
        closure_traces=closure_traces,
        summary=summary,
    )


def _build_summary(
    onehop_traces: list[RecurrentFrontierTrace],
    closure_traces: list[RecurrentFrontierTrace],
    config: RecurrentFrontierConfig,
) -> dict[str, Any]:
    rows = []
    for oh, cl in zip(onehop_traces, closure_traces):
        rows.append({
            "t_outer": oh.t_outer,
            "onehop_active_nodes": oh.final_active_node_count,
            "onehop_active_edges": oh.final_active_edge_count,
            "onehop_frontier_expansions": oh.total_frontier_expansions,
            "onehop_wall_ms": round(oh.wall_ms, 3),
            "closure_active_nodes": cl.final_active_node_count,
            "closure_active_edges": cl.final_active_edge_count,
            "closure_frontier_expansions": cl.total_frontier_expansions,
            "closure_path_count": cl.total_closure_path_count,
            "closure_compute_cost": cl.total_closure_compute_cost,
            "closure_wall_ms": round(cl.wall_ms, 3),
        })

    acceptance_met: bool | None = None
    if config.t_outer_values:
        t_min = min(config.t_outer_values)
        t_max = max(config.t_outer_values)
        oh_baseline = next((t for t in onehop_traces if t.t_outer == t_min), None)
        cl_best = next((t for t in closure_traces if t.t_outer == t_max), None)
        if oh_baseline is not None and cl_best is not None:
            acceptance_met = j_score(cl_best) > j_score(oh_baseline)

    return {
        "schema_version": SCHEMA_VERSION,
        "rows": rows,
        "acceptance_met": acceptance_met,
        "k": config.k,
        "l": config.l,
        "budget": config.budget,
        "h_horizon": config.h_horizon,
        "planner_pair": f"one_hop_topk vs {config.planner}",
        "note": "acceptance = J(T_outer=max,closure) > J(T_outer=min,one_hop_topk)",
    }
