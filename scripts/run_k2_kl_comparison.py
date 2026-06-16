#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.qwen_k2_kl_comparison import run_k2_kl_comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run KL distillation training on A_qwen k=2 vs matched random k=2."
    )
    parser.add_argument("--qwen-eval-dir", required=True,
                        help="v25 handoff dir containing the Qwen k=2 adjacency")
    parser.add_argument("--random-eval-dir", required=True,
                        help="Baseline comparison dir containing the random k=2 adjacency")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--random-adjacency-name", default=None,
                        help="Name of the random adjacency (default: qwen_topk_k<k>_random_seed0)")
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--forward-steps", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    report = run_k2_kl_comparison(
        args.qwen_eval_dir,
        args.random_eval_dir,
        args.output_dir,
        k=args.k,
        random_adjacency_name=args.random_adjacency_name,
        vocab_size=args.vocab_size,
        feature_dim=args.feature_dim,
        forward_steps=args.forward_steps,
        train_steps=args.train_steps,
        lr=args.lr,
        projection_seed=args.projection_seed,
        temperature=args.temperature,
        device=args.device,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
