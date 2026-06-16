#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v26_graph_decoder import CHECKPOINT_FILENAME, generate, load_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate from a trained v26 full graph decoder checkpoint.")
    parser.add_argument("--checkpoint", required=True, help=f"Path to {CHECKPOINT_FILENAME} or its parent dir")
    parser.add_argument("--prompt", required=True, help="Input prompt text")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compact", action="store_true", help="Print only generated text")
    args = parser.parse_args(argv)

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.is_dir():
        ckpt_path = ckpt_path / CHECKPOINT_FILENAME

    model, tokenizer = load_checkpoint(ckpt_path, device=args.device)
    result = generate(
        model,
        tokenizer,
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        device=args.device,
    )
    if args.compact:
        print(result["text"])
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
