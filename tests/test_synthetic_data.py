import json

import pytest

from src.normalize import normalize
from src.parser import parse
from src.shape import ShapeError, infer_shape
from src.synthetic_data import generate_hard_records, generate_records, generate_splits
from src.tasks import load_route_examples, load_shape_examples


def test_generate_records_is_deterministic():
    first = generate_records(20, seed=7, split="train")
    second = generate_records(20, seed=7, split="train")

    assert first == second
    assert any(rec["task"] == "route" for rec in first)
    assert any(rec["task"] == "shape_validity" for rec in first)


def test_generated_shape_records_match_shape_engine():
    records = generate_records(100, seed=11, split="train", route_fraction=0.0)

    assert any(not rec["valid"] for rec in records)
    for rec in records:
        env = {name: tuple(shape) for name, shape in rec["shape"].items() if name != "out"}
        node = normalize(parse(rec["expr"]))
        if rec["valid"]:
            out = infer_shape(node, env)
            assert list(out or ()) == rec["shape"].get("out", [])
        else:
            with pytest.raises(ShapeError):
                infer_shape(node, env)


def test_generate_splits_write_loadable_jsonl(tmp_path):
    manifest = generate_splits(tmp_path, train=12, val=8, test=6, seed=3)

    assert manifest["splits"]["train"]["records"] == 12
    assert (tmp_path / "manifest.json").exists()

    route_examples = load_route_examples(tmp_path / "train.jsonl")
    shape_examples = load_shape_examples(tmp_path / "train.jsonl")

    assert route_examples
    assert shape_examples
    assert any(not ex.valid for ex in shape_examples)


def test_generate_hard_records_has_many_unique_nested_routes():
    records = generate_hard_records(
        120,
        seed=19,
        split="hard",
        route_fraction=1.0,
        max_depth=4,
    )
    normalized = [rec["normalized"] for rec in records]

    assert len(set(normalized)) >= 100
    assert any(expr.count("(") >= 4 for expr in normalized)
    assert all(rec["task"] == "route" for rec in records)
    assert {rec["expert"] for rec in records} >= {
        "affine_expert",
        "matmul_expert",
        "reduction_expert",
        "grad_expert",
        "constraint_expert",
        "generic_expert",
    }


def test_generate_hard_splits_manifest_tracks_unique_routes(tmp_path):
    manifest = generate_splits(
        tmp_path,
        train=40,
        val=40,
        test=40,
        seed=23,
        route_fraction=1.0,
        hard=True,
        max_depth=4,
    )

    assert manifest["hard"] is True
    assert manifest["max_depth"] == 4
    assert manifest["splits"]["train"]["unique_route_expr"] >= 30
    assert (tmp_path / "train.jsonl").exists()


def test_shape_loader_preserves_valid_flag_without_mutating_shape(tmp_path):
    path = tmp_path / "examples.jsonl"
    rec = {
        "expr": "add(x, y)",
        "normalized": "add(x, y)",
        "shape": {"x": [4], "y": [8]},
        "valid": False,
    }
    path.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    first = load_shape_examples(path)
    second = load_shape_examples(path)

    assert first[0].valid is False
    assert second[0].valid is False
    assert first[0].env == {"x": (4,), "y": (8,)}
