#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v26_rewire_k2_cycle import run_v26_rewire_cycle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "v26 bounded rewiring cycle: proposal → KL gate → random gate → heldout gate.\n"
            "Baseline is qwen_topk_k2. Candidate is promoted only if all three gates pass."
        )
    )
    parser.add_argument("--qwen-eval-dir", required=True,
                        help="v25 handoff dir containing qwen_topk_k2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-artifacts", required=True,
                        help="Dir with distill_examples.jsonl (for heldout eval)")
    parser.add_argument("--baseline-k", type=int, default=2)
    parser.add_argument("--baseline-adjacency-name", default="qwen_topk_k2")
    parser.add_argument("--max-swaps", type=int, default=4)
    parser.add_argument("--proposal-policy", default="same_source_top_weight",
                        choices=["same_source_top_weight", "same_relation_top_weight",
                                 "utility_ratio", "same_source_low_weight", "deterministic_random"])
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=8)
    parser.add_argument("--forward-steps", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--held-out-per-family", type=int, default=8)
    parser.add_argument("--heldout-train-steps", type=int, default=128)
    parser.add_argument("--heldout-lr", type=float, default=0.5)
    parser.add_argument("--max-kl-regression", type=float, default=0.0)
    args = parser.parse_args(argv)

    report = run_v26_rewire_cycle(
        args.qwen_eval_dir,
        args.output_dir,
        args.teacher_artifacts,
        baseline_k=args.baseline_k,
        baseline_adjacency_name=args.baseline_adjacency_name,
        max_swaps=args.max_swaps,
        proposal_policy=args.proposal_policy,
        policy_seed=args.policy_seed,
        random_seed=args.random_seed,
        vocab_size=args.vocab_size,
        feature_dim=args.feature_dim,
        forward_steps=args.forward_steps,
        train_steps=args.train_steps,
        lr=args.lr,
        projection_seed=args.projection_seed,
        temperature=args.temperature,
        device=args.device,
        held_out_per_family=args.held_out_per_family,
        heldout_train_steps=args.heldout_train_steps,
        heldout_lr=args.heldout_lr,
        max_kl_regression=args.max_kl_regression,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
