#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v26_synthetic_data import generate_task_family_dataset, write_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic v26 synthetic task-family data.")
    parser.add_argument("--family", required=True, choices=["arithmetic_short", "symbolic_short"])
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    dataset = generate_task_family_dataset(args.family, args.n_train, args.n_eval, seed=args.seed)
    out = Path(args.output_dir)
    write_dataset(dataset["train"], out / "train.jsonl")
    write_dataset(dataset["eval"], out / "eval.jsonl")
    write_dataset(dataset["train"], out / "distill_examples.jsonl")
    print(json.dumps({
        "family": args.family,
        "n_train": len(dataset["train"]),
        "n_eval": len(dataset["eval"]),
        "output_dir": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
