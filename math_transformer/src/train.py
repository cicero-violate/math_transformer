from __future__ import annotations
import argparse
import json
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

import torch
import torch.nn as nn
from .model import MathRoutedTransformer
from .parser import parse
from .normalize import normalize
from .tasks import load_route_examples, route_loss, EXPERT_TO_ID


def _load_config(path: str) -> dict:
    with open(path) as f:
        if yaml is not None:
            return yaml.safe_load(f)
        raise ImportError("PyYAML is required. pip install pyyaml")


def resolve_data_path(config_path: str, relative_data_path: str) -> Path:
    """
    Resolve a data path relative to the project root.
    Project root is defined as config_file.parent.parent.
    This works regardless of the caller's working directory.
    """
    config_p = Path(config_path).resolve()
    project_root = config_p.parent.parent
    return project_root / relative_data_path


def _make_model(cfg: dict) -> MathRoutedTransformer:
    mc = cfg["model"]
    return MathRoutedTransformer(
        d_model=mc["d_model"],
        n_heads=mc["n_heads"],
        n_layers=mc["n_layers"],
        d_ff=mc["d_ff"],
        dropout=mc["dropout"],
        topk=mc["topk"],
        local_window=mc["local_window"],
    )


def train(config_path: str) -> None:
    cfg = _load_config(config_path)
    tc = cfg["training"]
    ec = cfg["eval"]
    data_path = resolve_data_path(config_path, cfg["data"]["path"])

    route_examples = load_route_examples(data_path)
    device = torch.device("cpu")
    model = _make_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Loaded {len(route_examples)} route examples from {data_path}")

    for step in range(tc["max_steps"]):
        model.train()
        ex = route_examples[step % len(route_examples)]
        root = normalize(parse(ex.expr))
        nodes = root.collect_nodes()

        x = model.embed_nodes(nodes).to(device)
        out, masks, _ = model(x, nodes)

        # Route prediction: predict expert at root token (position 0)
        logits = model.route_logits(out)  # (1, n_experts)
        loss = route_loss(logits, ex.expert_id)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % ec["interval"] == 0:
            pred_id = logits.argmax(dim=-1).item()
            correct = pred_id == ex.expert_id
            mask = masks[0]
            n = len(nodes)
            sparsity = mask.float().mean().item() if mask is not None else 1.0
            print(
                f"step={step:4d}  loss={loss.item():.4f}  "
                f"route={'✓' if correct else '✗'}({ex.expert})  "
                f"n={n:2d}  sparsity={sparsity:.3f}"
            )

    print("Training complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Math-Routed Transformer")
    parser.add_argument("--config", default="configs/tiny.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
