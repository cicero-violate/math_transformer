#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v26_graph_decoder import train_full_graph_decoder_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the v26 full graph decoder LM.")
    parser.add_argument("--adjacency-path", required=True, help="Path to selected adjacency JSON")
    parser.add_argument("--teacher-artifacts", required=True, help="Dir containing distill_examples.jsonl")
    parser.add_argument("--output-dir", required=True, help="Output dir for checkpoint + report")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--graph-bias-weight", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    result = train_full_graph_decoder_model(
        args.adjacency_path,
        args.teacher_artifacts,
        args.output_dir,
        block_size=args.block_size,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        graph_bias_weight=args.graph_bias_weight,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(result["report"], indent=2))
    print(f"\ncheckpoint: {result['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
