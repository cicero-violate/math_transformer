from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .eval import _load_route_eval_records
from .model import MathRoutedTransformer
from .normalize import normalize
from .parser import parse


def _resolve_device(device_name: str | None) -> torch.device:
    if device_name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return dev


def _load_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def export_dense_teacher_traces(
    *,
    examples_path: str,
    checkpoint: str,
    out_path: str,
    device: str | None = None,
    max_examples: int = 0,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 128,
) -> dict[str, int | str]:
    if not Path(examples_path).exists():
        raise FileNotFoundError(
            f"examples file not found: {examples_path}. "
            "Generate it with scripts/generate_hard_synthetic.sh or pass --examples to an existing JSONL."
        )
    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}. "
            "Train it with scripts/train_hard_synthetic.sh or pass --checkpoint to an existing checkpoint."
        )
    records = _load_route_eval_records(examples_path)
    if max_examples and max_examples > 0:
        records = records[:max_examples]
    dev = _resolve_device(device)
    model = MathRoutedTransformer(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=0.0,
        attention_mode="full",
        share_topology_cache=False,
    ).to(dev)
    state = _load_checkpoint(checkpoint, dev)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f, torch.no_grad():
        for rec in records:
            root = normalize(parse(rec["expr"]))
            nodes = root.collect_nodes()
            x = model.embed_nodes(nodes).to(dev)
            hidden = model(x, None, env=rec["env"] or None)[0]
            logits = model.route_logits(hidden)
            payload = {
                "expr": rec["expr"],
                "env": {k: list(v) for k, v in (rec["env"] or {}).items()},
                "expert": rec["expert"],
                "expert_id": int(rec["expert_id"]),
                "n_nodes": len(nodes),
                "dense_logits": logits.squeeze(0).detach().cpu().float().tolist(),
                "dense_root_hidden": hidden[0, 0].detach().cpu().float().tolist(),
            }
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    return {"examples": len(records), "out": str(out), "device": str(dev)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dense teacher traces for sparse topology training.")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=64, dest="d_model")
    parser.add_argument("--n-heads", type=int, default=4, dest="n_heads")
    parser.add_argument("--n-layers", type=int, default=2, dest="n_layers")
    parser.add_argument("--d-ff", type=int, default=128, dest="d_ff")
    args = parser.parse_args()
    summary = export_dense_teacher_traces(
        examples_path=args.examples,
        checkpoint=args.checkpoint,
        out_path=args.out,
        device=args.device,
        max_examples=args.max_examples,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()