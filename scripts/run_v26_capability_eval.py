#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v26_capability_eval import eval_capability, eval_gate_passes
from src.v26_graph_decoder import CHECKPOINT_FILENAME, load_checkpoint
from src.v26_synthetic_data import load_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a v26 decoder capability eval gate.")
    parser.add_argument("--checkpoint", required=True, help=f"Path to {CHECKPOINT_FILENAME} or its parent dir")
    parser.add_argument("--eval-data", required=True, help="Path to eval.jsonl")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", help="Optional path to write eval report JSON")
    args = parser.parse_args(argv)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / CHECKPOINT_FILENAME
    model, tokenizer = load_checkpoint(ckpt_path, device=args.device)
    eval_examples = load_dataset(args.eval_data)
    report = eval_capability(model, tokenizer, eval_examples, device=args.device)
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0 if eval_gate_passes(report, threshold=args.threshold) else 1


if __name__ == "__main__":
    raise SystemExit(main())
