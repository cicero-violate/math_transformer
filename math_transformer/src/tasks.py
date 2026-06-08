from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .router import EXPERT_NAMES

# ── Expert ID mapping ─────────────────────────────────────────────────────────

EXPERT_TO_ID: dict[str, int] = {name: i for i, name in enumerate(EXPERT_NAMES)}
ID_TO_EXPERT: dict[int, str] = {i: name for i, name in enumerate(EXPERT_NAMES)}
N_EXPERTS: int = len(EXPERT_NAMES)


# ── Route prediction task ─────────────────────────────────────────────────────

@dataclass
class RouteExample:
    expr: str          # normalized expression string
    expert: str        # target expert name
    expert_id: int     # integer label


def load_route_examples(path: str | Path) -> list[RouteExample]:
    """Load route-prediction examples from a JSONL file."""
    examples: list[RouteExample] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "expert" not in d:
                continue  # skip pairs/masks
            expr = d.get("normalized") or d.get("expr", "")
            expert = d["expert"]
            if expert not in EXPERT_TO_ID:
                continue
            examples.append(RouteExample(
                expr=expr,
                expert=expert,
                expert_id=EXPERT_TO_ID[expert],
            ))
    return examples


def route_loss(logits: torch.Tensor, target_id: int) -> torch.Tensor:
    """Cross-entropy loss for route prediction."""
    target = torch.tensor([target_id], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)


# ── Shape validity task ───────────────────────────────────────────────────────

@dataclass
class ShapeExample:
    expr: str
    env: dict[str, tuple[int, ...]]
    valid: bool
    output_shape: Optional[tuple[int, ...]]


def load_shape_examples(path: str | Path) -> list[ShapeExample]:
    """Load shape-validity examples from JSONL (requires 'shape' field)."""
    examples: list[ShapeExample] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "shape" not in d or not isinstance(d["shape"], dict):
                continue
            expr = d.get("normalized") or d.get("expr", "")
            shape_dict = dict(d["shape"])
            out_raw = shape_dict.pop("out", None)
            out_shape = tuple(out_raw) if out_raw else None
            env = {k: tuple(v) for k, v in shape_dict.items()}
            examples.append(ShapeExample(
                expr=expr,
                env=env,
                valid=bool(d.get("valid", True)),
                output_shape=out_shape,
            ))
    return examples
