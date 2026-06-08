from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .eval import run_quality_eval
from .topology import RELATION_WEIGHTS


SEARCH_KEYS = [
    "symbolic_dependency",
    "composition",
    "shape_compat",
    "embedding",
    "local_window",
    "same_operator",
]


@dataclass
class SearchResult:
    iteration: int
    objective: float
    route_accuracy: float
    dense_agreement: float
    k: int
    weights: dict[str, float]


def _score(route_accuracy: float, dense_agreement: float, quality_weight: float) -> float:
    return quality_weight * route_accuracy + (1.0 - quality_weight) * dense_agreement


def _mutate_weights(
    base: dict[str, float],
    rng: random.Random,
    step: float,
    min_weight: float,
    max_weight: float,
) -> dict[str, float]:
    out = dict(base)
    out["identity"] = RELATION_WEIGHTS["identity"]
    for key in SEARCH_KEYS:
        current = float(out.get(key, RELATION_WEIGHTS[key]))
        factor = 1.0 + rng.uniform(-step, step)
        out[key] = min(max(current * factor, min_weight), max_weight)
    return out


def _evaluate_weights(
    weights: dict[str, float],
    examples_path: str,
    checkpoint: str | None,
    k: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    topk: int,
    local_window: int,
    device: str | None,
    quality_weight: float,
) -> tuple[float, float, float]:
    reports = run_quality_eval(
        examples_path=examples_path,
        k_values=[k],
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        topk=topk,
        local_window=local_window,
        checkpoint=checkpoint,
        device=device,
        topology_mode="scored_topk",
        fixed_k=k,
        relation_weights=weights,
    )
    sparse = reports[1]
    dense_agreement = float(sparse.dense_agreement or 0.0)
    route_accuracy = float(sparse.route_accuracy)
    return _score(route_accuracy, dense_agreement, quality_weight), route_accuracy, dense_agreement


def search_topology(
    examples_path: str,
    checkpoint: str | None,
    output: str,
    iterations: int,
    forever: bool = False,
    k: int = 16,
    seed: int = 0,
    step: float = 0.35,
    min_weight: float = 0.0,
    max_weight: float = 4.0,
    quality_weight: float = 0.75,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 128,
    topk: int = 3,
    local_window: int = 1,
    device: str | None = None,
) -> SearchResult:
    rng = random.Random(seed)
    current = dict(RELATION_WEIGHTS)
    best_obj, best_acc, best_agree = _evaluate_weights(
        current,
        examples_path,
        checkpoint,
        k,
        d_model,
        n_heads,
        n_layers,
        d_ff,
        topk,
        local_window,
        device,
        quality_weight,
    )
    best = SearchResult(
        iteration=0,
        objective=best_obj,
        route_accuracy=best_acc,
        dense_agreement=best_agree,
        k=k,
        weights=current,
    )
    _write_result(output, best)
    print(_format_result("initial", best))

    i = 0
    while forever or i < iterations:
        i += 1
        candidate = _mutate_weights(best.weights, rng, step, min_weight, max_weight)
        obj, acc, agree = _evaluate_weights(
            candidate,
            examples_path,
            checkpoint,
            k,
            d_model,
            n_heads,
            n_layers,
            d_ff,
            topk,
            local_window,
            device,
            quality_weight,
        )
        if obj >= best.objective:
            best = SearchResult(
                iteration=i,
                objective=obj,
                route_accuracy=acc,
                dense_agreement=agree,
                k=k,
                weights=candidate,
            )
            _write_result(output, best)
            print(_format_result("accepted", best))
        else:
            print(
                f"rejected iteration={i} objective={obj:.4f} "
                f"route_acc={acc:.4f} dense_agree={agree:.4f}"
            )
        if forever:
            time.sleep(0.01)
    return best


def _write_result(output: str, result: SearchResult) -> None:
    p = Path(output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")


def _format_result(label: str, result: SearchResult) -> str:
    return (
        f"{label} iteration={result.iteration} objective={result.objective:.4f} "
        f"route_acc={result.route_accuracy:.4f} dense_agree={result.dense_agreement:.4f} "
        f"weights={json.dumps(result.weights, sort_keys=True)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline topology relation-weight search")
    parser.add_argument("--examples", default="data/examples.jsonl")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default="runs/topology_search/best_weights.json")
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--forever", action="store_true")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=float, default=0.35)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float, default=4.0)
    parser.add_argument("--quality-weight", type=float, default=0.75)
    parser.add_argument("--d-model", type=int, default=64, dest="d_model")
    parser.add_argument("--n-heads", type=int, default=4, dest="n_heads")
    parser.add_argument("--n-layers", type=int, default=2, dest="n_layers")
    parser.add_argument("--d-ff", type=int, default=128, dest="d_ff")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--local-window", type=int, default=1, dest="local_window")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    kwargs = vars(args)
    kwargs["examples_path"] = kwargs.pop("examples")
    search_topology(**kwargs)


if __name__ == "__main__":
    main()
