from __future__ import annotations
import argparse
import csv
import json
import time
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


def _make_model(
    cfg: dict,
    *,
    attention_mode_override: str | None = None,
    max_neighbors_override: int | None = None,
    topology_mode_override: str | None = None,
    fixed_k_override: int | None = None,
    middle_bridge_width_override: int | None = None,
) -> MathRoutedTransformer:
    mc = cfg["model"]
    attention_mode = attention_mode_override or mc.get("attention_mode", "dense_masked")
    topology_mode = topology_mode_override or mc.get("topology_mode", "union")
    fixed_k = fixed_k_override if fixed_k_override is not None else mc.get("fixed_k", 32)
    max_neighbors = max_neighbors_override if max_neighbors_override is not None else mc.get("max_neighbors", 16)
    middle_bridge_width = (
        middle_bridge_width_override
        if middle_bridge_width_override is not None
        else mc.get("middle_bridge_width", 0)
    )
    return MathRoutedTransformer(
        d_model=mc["d_model"],
        n_heads=mc["n_heads"],
        n_layers=mc["n_layers"],
        d_ff=mc["d_ff"],
        dropout=mc["dropout"],
        topk=mc["topk"],
        local_window=mc["local_window"],
        attention_mode=attention_mode,
        max_neighbors=max_neighbors,
        topology_mode=topology_mode,
        fixed_k=fixed_k,
        middle_bridge_width=middle_bridge_width,
    )


def _resolve_device(device_name: str | None) -> torch.device:
    if device_name is None:
        return torch.device("cpu")
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return dev


def _cache_stats(model: MathRoutedTransformer) -> tuple[int, int]:
    hits = 0
    misses = 0
    seen: set[int] = set()
    for layer in getattr(model, "layers", []):
        cache = getattr(layer, "_topology_cache", None)
        if cache is None or id(cache) in seen:
            continue
        seen.add(id(cache))
        hits += int(getattr(cache, "cache_hits", 0))
        misses += int(getattr(cache, "cache_misses", 0))
    return hits, misses


def train(
    config_path: str,
    save_checkpoint: str | None = None,
    data_path_override: str | None = None,
    max_steps_override: int | None = None,
    eval_interval_override: int | None = None,
    attention_mode_override: str | None = None,
    topology_mode_override: str | None = None,
    fixed_k_override: int | None = None,
    max_neighbors_override: int | None = None,
    middle_bridge_width_override: int | None = None,
    device_override: str | None = None,
    save_loss_csv: str | None = None,
) -> None:
    cfg = _load_config(config_path)
    tc = cfg["training"]
    ec = cfg["eval"]
    if max_steps_override is not None:
        tc["max_steps"] = max_steps_override
    if eval_interval_override is not None:
        ec["interval"] = eval_interval_override
    data_path = (
        Path(data_path_override)
        if data_path_override
        else resolve_data_path(config_path, cfg["data"]["path"])
    )

    route_examples = load_route_examples(data_path)
    device = _resolve_device(device_override or cfg.get("training", {}).get("device"))
    model = _make_model(
        cfg,
        attention_mode_override=attention_mode_override,
        max_neighbors_override=max_neighbors_override,
        topology_mode_override=topology_mode_override,
        fixed_k_override=fixed_k_override,
        middle_bridge_width_override=middle_bridge_width_override,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"])

    first_layer = model.layers[0] if len(model.layers) else None
    active_topology = getattr(first_layer, "topology", None) if first_layer is not None else None
    active_attention_mode = getattr(model, "attention_mode", "unknown")
    active_topology_mode = getattr(active_topology, "topology_mode", "full")
    active_fixed_k = getattr(active_topology, "fixed_k", None)
    active_middle_bridge_width = getattr(active_topology, "middle_bridge_width", None)
    active_max_neighbors = max_neighbors_override if max_neighbors_override is not None else cfg["model"].get("max_neighbors", 16)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Loaded {len(route_examples)} route examples from {data_path}")
    print(
        "Training route: "
        f"device={device}  attention_mode={active_attention_mode}  "
        f"topology_mode={active_topology_mode}  fixed_k={active_fixed_k}  "
        f"max_neighbors={active_max_neighbors}  "
        f"middle_bridge_width={active_middle_bridge_width}"
    )

    csv_file = None
    csv_writer = None
    if save_loss_csv:
        csv_path = Path(save_loss_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "step", "loss", "correct", "expert", "pred_id", "target_id",
                "n_nodes", "sparsity", "step_ms", "device", "attention_mode",
                "topology_mode", "fixed_k", "max_neighbors", "middle_bridge_width",
                "cache_hits", "cache_misses",
            ],
        )
        csv_writer.writeheader()

    try:
        for step in range(tc["max_steps"]):
            model.train()
            ex = route_examples[step % len(route_examples)]
            root = normalize(parse(ex.expr))
            nodes = root.collect_nodes()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            x = model.embed_nodes(nodes).to(device)
            out, masks, _ = model(x, nodes)

            # Route prediction: predict expert at root token (position 0)
            logits = model.route_logits(out)  # (1, n_experts)
            loss = route_loss(logits, ex.expert_id)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_ms = (time.perf_counter() - t0) * 1000.0

            pred_id = logits.argmax(dim=-1).item()
            correct = pred_id == ex.expert_id
            mask = masks[0]
            n = len(nodes)
            sparsity = mask.float().mean().item() if mask is not None else 1.0
            cache_hits, cache_misses = _cache_stats(model)

            if csv_writer is not None:
                csv_writer.writerow({
                    "step": step,
                    "loss": float(loss.item()),
                    "correct": int(correct),
                    "expert": ex.expert,
                    "pred_id": int(pred_id),
                    "target_id": int(ex.expert_id),
                    "n_nodes": n,
                    "sparsity": float(sparsity),
                    "step_ms": float(step_ms),
                    "device": str(device),
                    "attention_mode": active_attention_mode,
                    "topology_mode": active_topology_mode,
                    "fixed_k": active_fixed_k,
                    "max_neighbors": active_max_neighbors,
                    "middle_bridge_width": active_middle_bridge_width,
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                })

            if step % ec["interval"] == 0:
                print(
                    f"step={step:4d}  loss={loss.item():.4f}  "
                    f"route={'✓' if correct else '✗'}({ex.expert})  "
                    f"n={n:2d}  sparsity={sparsity:.3f}  "
                    f"step_ms={step_ms:.3f}"
                )
    finally:
        if csv_file is not None:
            csv_file.close()

    print("Training complete.")
    if save_checkpoint:
        p = Path(save_checkpoint)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": cfg,
            },
            p,
        )
        print(f"Saved checkpoint to {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Math-Routed Transformer")
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--save-checkpoint", default=None, dest="save_checkpoint")
    parser.add_argument("--data", default=None, dest="data_path_override")
    parser.add_argument("--max-steps", type=int, default=None, dest="max_steps_override")
    parser.add_argument("--eval-interval", type=int, default=None, dest="eval_interval_override")
    parser.add_argument(
        "--attention-mode",
        default=None,
        choices=["full", "dense_masked", "neighbor_sparse"],
        dest="attention_mode_override",
    )
    parser.add_argument(
        "--topology-mode",
        default=None,
        choices=["union", "scored_topk", "middle_preserving_topk"],
        dest="topology_mode_override",
    )
    parser.add_argument("--fixed-k", type=int, default=None, dest="fixed_k_override")
    parser.add_argument("--max-neighbors", type=int, default=None, dest="max_neighbors_override")
    parser.add_argument("--middle-bridge-width", type=int, default=None, dest="middle_bridge_width_override")
    parser.add_argument("--device", default=None, dest="device_override")
    parser.add_argument("--save-loss-csv", default=None, dest="save_loss_csv")
    args = parser.parse_args()
    train(
        args.config,
        save_checkpoint=args.save_checkpoint,
        data_path_override=args.data_path_override,
        max_steps_override=args.max_steps_override,
        eval_interval_override=args.eval_interval_override,
        attention_mode_override=args.attention_mode_override,
        topology_mode_override=args.topology_mode_override,
        fixed_k_override=args.fixed_k_override,
        max_neighbors_override=args.max_neighbors_override,
        middle_bridge_width_override=args.middle_bridge_width_override,
        device_override=args.device_override,
        save_loss_csv=args.save_loss_csv,
    )


if __name__ == "__main__":
    main()
