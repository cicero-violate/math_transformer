#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v26_graph_ar import train_graph_ar_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the graph-autoregressive LM on distill_examples.jsonl."
    )
    parser.add_argument("--adjacency-path", required=True, help="Path to selected adjacency JSON (qwen_topk_k2.json)")
    parser.add_argument("--teacher-artifacts", required=True, help="Dir containing distill_examples.jsonl")
    parser.add_argument("--output-dir", required=True, help="Output dir for checkpoint + report")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    result = train_graph_ar_model(
        args.adjacency_path,
        args.teacher_artifacts,
        args.output_dir,
        hidden_dim=args.hidden_dim,
        n_steps=args.n_steps,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
    )
    print(json.dumps(result["report"], indent=2))
    print(f"\ncheckpoint: {result['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
