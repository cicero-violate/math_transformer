#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.qwen_random_adjacency_baseline import run_kl_baseline_comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run KL eval on A_qwen vs matched random adjacency baseline."
    )
    parser.add_argument("--source-eval-dir", required=True,
                        help="v25 handoff dir containing the Qwen adjacency")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write comparison artifacts")
    parser.add_argument("--k", type=int, default=None,
                        help="Select adjacency by k (default: first in index)")
    parser.add_argument("--adjacency-name", default=None,
                        help="Select adjacency by name (alternative to --k)")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--forward-steps", type=int, default=1)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    report = run_kl_baseline_comparison(
        args.source_eval_dir,
        args.output_dir,
        k=args.k,
        adjacency_name=args.adjacency_name,
        random_seed=args.random_seed,
        vocab_size=args.vocab_size,
        feature_dim=args.feature_dim,
        forward_steps=args.forward_steps,
        projection_seed=args.projection_seed,
        temperature=args.temperature,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
