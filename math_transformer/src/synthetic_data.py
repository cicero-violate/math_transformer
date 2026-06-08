from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .normalize import normalize
from .parser import parse
from .router import OperatorRouter
from .shape import ShapeError, infer_shape


Shape = tuple[int, ...]
Record = dict[str, Any]

_DIMS = (4, 8, 16, 32, 64, 128)


def _norm(expr: str) -> str:
    return repr(normalize(parse(expr)))


def _shape_dict(env: dict[str, Shape], out: Shape | None = None) -> dict[str, list[int]]:
    data = {name: list(shape) for name, shape in env.items()}
    if out is not None:
        data["out"] = list(out)
    return data


def _expert_for(expr: str) -> str:
    return OperatorRouter().route(normalize(parse(expr))).expert


def _route_record(split: str, idx: int, expr: str, env: dict[str, Shape]) -> Record:
    node = normalize(parse(expr))
    out = infer_shape(node, env)
    return {
        "id": f"{split}-route-{idx:06d}",
        "task": "route",
        "expr": expr,
        "normalized": repr(node),
        "expert": _expert_for(expr),
        "shape": _shape_dict(env, out),
        "valid": True,
    }


def _shape_record(
    split: str,
    idx: int,
    expr: str,
    env: dict[str, Shape],
    valid: bool,
) -> Record:
    out: Shape | None = None
    if valid:
        out = infer_shape(normalize(parse(expr)), env)
    return {
        "id": f"{split}-shape-{idx:06d}",
        "task": "shape_validity",
        "expr": expr,
        "normalized": _norm(expr),
        "shape": _shape_dict(env, out),
        "valid": valid,
    }


def _dim(rng: random.Random) -> int:
    return rng.choice(_DIMS)


def _different_dim(rng: random.Random, value: int) -> int:
    choices = [d for d in _DIMS if d != value]
    return rng.choice(choices)


def _valid_route(rng: random.Random, split: str, idx: int) -> Record:
    kind = rng.choice(("affine", "matmul", "elementwise", "reduction", "grad", "constraint"))
    if kind == "affine":
        m, k = _dim(rng), _dim(rng)
        return _route_record(split, idx, "affine(A, x, b)", {"A": (m, k), "x": (k,), "b": (m,)})
    if kind == "matmul":
        m, k, n = _dim(rng), _dim(rng), _dim(rng)
        if rng.random() < 0.5:
            return _route_record(split, idx, "matmul(A, x)", {"A": (m, k), "x": (k,)})
        return _route_record(split, idx, "matmul(A, B)", {"A": (m, k), "B": (k, n)})
    if kind == "elementwise":
        d = _dim(rng)
        expr = rng.choice(("add(x, y)", "sub(x, y)", "mul(x, y)", "div(x, y)"))
        return _route_record(split, idx, expr, {"x": (d,), "y": (d,)})
    if kind == "reduction":
        d = _dim(rng)
        expr = rng.choice(("sum(x)", "mean(x)", "norm(x)"))
        return _route_record(split, idx, expr, {"x": (d,)})
    if kind == "grad":
        d = _dim(rng)
        expr = rng.choice(("grad(sum(x), x)", "grad(norm(x), x)", "grad(add(x, y), x)"))
        env = {"x": (d,), "y": (d,)}
        return _route_record(split, idx, expr, env)

    m, k = _dim(rng), _dim(rng)
    return _route_record(
        split,
        idx,
        "constraint(leq(matmul(A, x), b))",
        {"A": (m, k), "x": (k,), "b": (m,)},
    )


def _shape_case(rng: random.Random, split: str, idx: int) -> Record:
    valid = rng.random() >= 0.4
    kind = rng.choice(("add", "matmul_vec", "matmul_mat", "affine"))

    if kind == "add":
        d = _dim(rng)
        y = d if valid else _different_dim(rng, d)
        return _shape_record(split, idx, "add(x, y)", {"x": (d,), "y": (y,)}, valid)

    if kind == "matmul_vec":
        m, k = _dim(rng), _dim(rng)
        x = k if valid else _different_dim(rng, k)
        return _shape_record(split, idx, "matmul(A, x)", {"A": (m, k), "x": (x,)}, valid)

    if kind == "matmul_mat":
        m, k, n = _dim(rng), _dim(rng), _dim(rng)
        inner = k if valid else _different_dim(rng, k)
        return _shape_record(split, idx, "matmul(A, B)", {"A": (m, k), "B": (inner, n)}, valid)

    m, k = _dim(rng), _dim(rng)
    b = m if valid else _different_dim(rng, m)
    return _shape_record(split, idx, "affine(A, x, b)", {"A": (m, k), "x": (k,), "b": (b,)}, valid)


def generate_records(count: int, *, seed: int, split: str, route_fraction: float = 0.6) -> list[Record]:
    rng = random.Random(seed)
    records: list[Record] = []
    for idx in range(count):
        if rng.random() < route_fraction:
            records.append(_valid_route(rng, split, idx))
        else:
            records.append(_shape_case(rng, split, idx))
    return records


def write_jsonl(path: Path, records: list[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def generate_splits(
    out_dir: Path,
    *,
    train: int,
    val: int,
    test: int,
    seed: int,
    route_fraction: float = 0.6,
) -> dict[str, Any]:
    split_counts = {"train": train, "val": val, "test": test}
    manifest: dict[str, Any] = {
        "seed": seed,
        "route_fraction": route_fraction,
        "splits": {},
    }
    for offset, (split, count) in enumerate(split_counts.items()):
        records = generate_records(
            count,
            seed=seed + offset * 100_003,
            split=split,
            route_fraction=route_fraction,
        )
        path = out_dir / f"{split}.jsonl"
        write_jsonl(path, records)
        manifest["splits"][split] = {
            "path": str(path),
            "records": len(records),
            "route_records": sum(1 for r in records if r.get("task") == "route"),
            "shape_records": sum(1 for r in records if r.get("task") == "shape_validity"),
            "invalid_shape_records": sum(1 for r in records if r.get("task") == "shape_validity" and not r.get("valid", True)),
        }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic math-transformer JSONL splits.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--train", type=int, default=10_000)
    parser.add_argument("--val", type=int, default=1_000)
    parser.add_argument("--test", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--route-fraction", type=float, default=0.6)
    args = parser.parse_args()

    manifest = generate_splits(
        args.out_dir,
        train=args.train,
        val=args.val,
        test=args.test,
        seed=args.seed,
        route_fraction=args.route_fraction,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
