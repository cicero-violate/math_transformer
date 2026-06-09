import json
from pathlib import Path

from src.topology import RELATION_WEIGHTS
from src.topology_search import _mutate_weights, search_topology


PROJ = Path(__file__).resolve().parents[1]


def test_mutate_weights_keeps_identity_fixed():
    import random

    rng = random.Random(0)
    mutated = _mutate_weights(
        RELATION_WEIGHTS,
        rng,
        step=0.5,
        min_weight=0.0,
        max_weight=4.0,
    )

    assert mutated["identity"] == RELATION_WEIGHTS["identity"]
    assert set(mutated) == set(RELATION_WEIGHTS)


def test_search_topology_writes_best_weights(tmp_path):
    output = tmp_path / "best.json"

    result = search_topology(
        examples_path=str(PROJ / "data" / "examples.jsonl"),
        checkpoint=None,
        output=str(output),
        iterations=1,
        k=2,
        seed=1,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        topk=1,
        local_window=1,
        device="cpu",
    )

    assert output.exists()
    data = json.loads(output.read_text())
    assert data["objective"] == result.objective
    assert data["k"] == 2
    assert "weights" in data
