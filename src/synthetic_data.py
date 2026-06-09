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
_VARS = tuple("xyzuvwabcdefghijklmnop")


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


def _fresh_name(rng: random.Random, used: set[str]) -> str:
    while True:
        base = rng.choice(_VARS)
        suffix = rng.randrange(10_000)
        name = f"{base}{suffix}"
        if name not in used:
            used.add(name)
            return name


def _hard_vector(rng: random.Random, env: dict[str, Shape], used: set[str], dim: int) -> str:
    name = _fresh_name(rng, used)
    env[name] = (dim,)
    return name


def _hard_matrix(rng: random.Random, env: dict[str, Shape], used: set[str], rows: int, cols: int) -> str:
    name = _fresh_name(rng, used).upper()
    env[name] = (rows, cols)
    return name


def _hard_vec_expr(
    rng: random.Random,
    env: dict[str, Shape],
    used: set[str],
    dim: int,
    depth: int,
) -> str:
    if depth <= 0:
        return _hard_vector(rng, env, used, dim)
    choice = rng.choice(("leaf", "add", "sub", "mul", "matmul", "affine", "grad", "scale"))
    if choice == "leaf":
        return _hard_vector(rng, env, used, dim)
    if choice in ("add", "sub", "mul"):
        a = _hard_vec_expr(rng, env, used, dim, depth - 1)
        b = _hard_vec_expr(rng, env, used, dim, depth - 1)
        return f"{choice}({a}, {b})"
    if choice == "scale":
        a = _hard_vec_expr(rng, env, used, dim, depth - 1)
        b = _hard_vector(rng, env, used, dim)
        return f"div(mul({a}, {b}), add({b}, {b}))"
    if choice == "matmul":
        inner = _dim(rng)
        m = _hard_matrix(rng, env, used, dim, inner)
        x = _hard_vec_expr(rng, env, used, inner, depth - 1)
        return f"matmul({m}, {x})"
    if choice == "affine":
        inner = _dim(rng)
        m = _hard_matrix(rng, env, used, dim, inner)
        x = _hard_vec_expr(rng, env, used, inner, depth - 1)
        b = _hard_vector(rng, env, used, dim)
        return f"affine({m}, {x}, {b})"
    x = _hard_vector(rng, env, used, dim)
    scalar = rng.choice((f"sum({x})", f"norm({x})", f"mean({x})"))
    return f"grad({scalar}, {x})"


def _hard_route(rng: random.Random, split: str, idx: int, max_depth: int) -> Record:
    env: dict[str, Shape] = {}
    used: set[str] = set()
    dim = _dim(rng)
    depth = rng.randint(2, max(2, max_depth))
    family = rng.choice(("affine", "matmul", "elementwise", "reduction", "grad", "constraint"))

    if family == "affine":
        inner = _dim(rng)
        a = _hard_matrix(rng, env, used, dim, inner)
        x = _hard_vec_expr(rng, env, used, inner, depth - 1)
        b = _hard_vec_expr(rng, env, used, dim, max(0, depth - 2))
        expr = f"affine({a}, {x}, {b})"
    elif family == "matmul":
        inner = _dim(rng)
        a = _hard_matrix(rng, env, used, dim, inner)
        x = _hard_vec_expr(rng, env, used, inner, depth - 1)
        expr = f"matmul({a}, {x})"
    elif family == "elementwise":
        op = rng.choice(("add", "sub", "mul", "div"))
        lhs = _hard_vec_expr(rng, env, used, dim, depth - 1)
        rhs = _hard_vec_expr(rng, env, used, dim, depth - 1)
        expr = f"{op}({lhs}, {rhs})"
    elif family == "reduction":
        op = rng.choice(("sum", "mean", "norm"))
        x = _hard_vec_expr(rng, env, used, dim, depth - 1)
        expr = f"{op}({x})"
    elif family == "grad":
        x = _hard_vector(rng, env, used, dim)
        body = rng.choice((f"sum({x})", f"norm({x})", f"mean({x})"))
        expr = f"grad({body}, {x})"
    else:
        inner = _dim(rng)
        a = _hard_matrix(rng, env, used, dim, inner)
        x = _hard_vec_expr(rng, env, used, inner, depth - 1)
        b = _hard_vec_expr(rng, env, used, dim, max(0, depth - 2))
        expr = f"constraint(leq(matmul({a}, {x}), {b}))"
    return _route_record(split, idx, expr, env)


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


def generate_hard_records(
    count: int,
    *,
    seed: int,
    split: str,
    route_fraction: float = 0.8,
    max_depth: int = 5,
) -> list[Record]:
    rng = random.Random(seed)
    records: list[Record] = []
    for idx in range(count):
        if rng.random() < route_fraction:
            records.append(_hard_route(rng, split, idx, max_depth=max_depth))
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
    hard: bool = False,
    max_depth: int = 5,
) -> dict[str, Any]:
    split_counts = {"train": train, "val": val, "test": test}
    manifest: dict[str, Any] = {
        "seed": seed,
        "route_fraction": route_fraction,
        "hard": hard,
        "max_depth": max_depth,
        "splits": {},
    }
    for offset, (split, count) in enumerate(split_counts.items()):
        generator = generate_hard_records if hard else generate_records
        kwargs = {"max_depth": max_depth} if hard else {}
        records = generator(
            count,
            seed=seed + offset * 100_003,
            split=split,
            route_fraction=route_fraction,
            **kwargs,
        )
        path = out_dir / f"{split}.jsonl"
        write_jsonl(path, records)
        manifest["splits"][split] = {
            "path": str(path),
            "records": len(records),
            "route_records": sum(1 for r in records if r.get("task") == "route"),
            "shape_records": sum(1 for r in records if r.get("task") == "shape_validity"),
            "invalid_shape_records": sum(1 for r in records if r.get("task") == "shape_validity" and not r.get("valid", True)),
            "unique_route_expr": len({r.get("normalized") for r in records if r.get("task") == "route"}),
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
    parser.add_argument("--hard", action="store_true", help="Generate deeper/harder route expressions.")
    parser.add_argument("--max-depth", type=int, default=5)
    args = parser.parse_args()

    manifest = generate_splits(
        args.out_dir,
        train=args.train,
        val=args.val,
        test=args.test,
        seed=args.seed,
        route_fraction=args.route_fraction,
        hard=args.hard,
        max_depth=args.max_depth,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
