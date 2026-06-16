#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v25_01_heldout_eval import run_v25_01_heldout_eval


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="v25.01 heldout text/task evaluation — stratified train/held-out split."
    )
    parser.add_argument("--teacher-artifacts", required=True,
                        help="Directory containing distill_examples.jsonl")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write heldout_eval_report.json")
    parser.add_argument("--held-out-per-family", type=int, default=8,
                        help="Number of examples per family to hold out (default: 8)")
    parser.add_argument("--train-steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=0)
    args = parser.parse_args(argv)

    report = run_v25_01_heldout_eval(
        args.teacher_artifacts,
        args.output_dir,
        held_out_per_family=args.held_out_per_family,
        train_steps=args.train_steps,
        lr=args.lr,
        split_seed=args.split_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
