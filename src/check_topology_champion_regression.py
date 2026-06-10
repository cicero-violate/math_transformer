from __future__ import annotations

import argparse
import json
from pathlib import Path

from .promote_topology_scorer import champion_regression_from_artifact, metrics_from_champion_metadata
from .topology_benchmark_artifact import load_artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify champion learned-topology scorer gates from a structured benchmark artifact."
    )
    parser.add_argument("--artifact", required=True, help="Benchmark artifact JSON or JSONL from benchmark_learned_topology.sh.")
    parser.add_argument("--champion-metadata", default="runs/checkpoints/topology_scorer.champion.json")
    parser.add_argument("--route-min-delta", type=float, default=0.0)
    parser.add_argument("--generic-min-delta", type=float, default=0.0)
    args = parser.parse_args()

    champion = metrics_from_champion_metadata(args.champion_metadata)
    if champion is None:
        raise SystemExit(f"champion metadata not found: {args.champion_metadata}")

    artifact = load_artifact(args.artifact)
    decision = champion_regression_from_artifact(
        artifact=artifact,
        champion=champion,
        route_min_delta=args.route_min_delta,
        generic_min_delta=args.generic_min_delta,
    )
    payload = {
        "regression_passed": bool(decision.promote),
        "reason": decision.reason,
        "candidate": None if decision.candidate is None else decision.candidate.__dict__,
        "champion": None if decision.champion is None else decision.champion.__dict__,
        "benchmark": None if decision.benchmark is None else decision.benchmark.__dict__,
        "artifact": str(Path(args.artifact)),
        "champion_metadata": str(Path(args.champion_metadata)),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not decision.promote:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
