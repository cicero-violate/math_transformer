from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest

from src.v25_01_heldout_eval import (
    HELDOUT_EVAL_REPORT_FILENAME,
    SCHEMA_VERSION,
    SPLIT_MANIFEST_FILENAME,
    _per_family_heldout_loss,
    _softmax_loss,
    _token_counts,
    _train_logit_bias,
    run_v25_01_heldout_eval,
    stratified_split,
)


# ---------------------------------------------------------------------------
# stratified_split
# ---------------------------------------------------------------------------

def _make_examples(families: list[str], n_per_family: int) -> list[dict]:
    examples = []
    for fam in families:
        for i in range(n_per_family):
            examples.append(
                {
                    "sample_id": f"{fam}_{i}",
                    "family": fam,
                    "target": f"answer: {i}",
                }
            )
    return examples


def test_stratified_split_counts():
    examples = _make_examples(["a", "b", "c"], 10)
    train, heldout = stratified_split(examples, held_out_per_family=3, seed=0)
    assert len(heldout) == 9   # 3 families × 3
    assert len(train) == 21    # 3 families × 7


def test_stratified_split_no_family_leakage():
    examples = _make_examples(["x", "y"], 5)
    train, heldout = stratified_split(examples, held_out_per_family=2, seed=42)
    held_ids = {ex["sample_id"] for ex in heldout}
    train_ids = {ex["sample_id"] for ex in train}
    assert held_ids.isdisjoint(train_ids)
    assert held_ids | train_ids == {ex["sample_id"] for ex in examples}


def test_stratified_split_clamps_to_family_size():
    examples = _make_examples(["small"], 2)
    train, heldout = stratified_split(examples, held_out_per_family=99, seed=0)
    assert len(heldout) == 2
    assert len(train) == 0


def test_stratified_split_deterministic():
    examples = _make_examples(["a", "b"], 20)
    t1, h1 = stratified_split(examples, held_out_per_family=5, seed=7)
    t2, h2 = stratified_split(examples, held_out_per_family=5, seed=7)
    assert [ex["sample_id"] for ex in h1] == [ex["sample_id"] for ex in h2]


# ---------------------------------------------------------------------------
# _softmax_loss
# ---------------------------------------------------------------------------

def test_softmax_loss_uniform():
    logits = {"a": 0.0, "b": 0.0}
    counts: Counter[str] = Counter({"a": 1, "b": 1})
    loss = _softmax_loss(logits, counts)
    assert math.isfinite(loss)
    assert abs(loss - math.log(2)) < 1e-9


def test_softmax_loss_zero_counts():
    logits = {"a": 0.0}
    assert math.isnan(_softmax_loss(logits, Counter()))


def test_softmax_loss_skips_oov_tokens():
    logits = {"a": 0.0, "b": 0.0}
    counts: Counter[str] = Counter({"a": 1, "UNSEEN": 1})
    loss = _softmax_loss(logits, counts)
    assert math.isfinite(loss)


# ---------------------------------------------------------------------------
# _train_logit_bias
# ---------------------------------------------------------------------------

def test_train_logit_bias_decreases_loss():
    counts: Counter[str] = Counter({"yes": 10, "no": 2, "maybe": 1})
    vocab = sorted(counts)
    initial_loss = _softmax_loss({tok: 0.0 for tok in vocab}, counts)
    trained = _train_logit_bias(counts, train_steps=32, lr=0.5)
    final_loss = _softmax_loss(trained, counts)
    assert final_loss < initial_loss


def test_train_logit_bias_returns_all_vocab():
    counts: Counter[str] = Counter({"x": 3, "y": 2})
    trained = _train_logit_bias(counts, train_steps=5, lr=0.1)
    assert set(trained) == {"x", "y"}


# ---------------------------------------------------------------------------
# run_v25_01_heldout_eval
# ---------------------------------------------------------------------------

def _write_distill_examples(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")


def test_run_heldout_eval_basic(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    examples = _make_examples(["arithmetic_short", "logic_short", "symbolic_short", "project_specific"], 16)
    _write_distill_examples(artifacts_dir / "distill_examples.jsonl", examples)

    report = run_v25_01_heldout_eval(
        artifacts_dir,
        tmp_path / "out",
        held_out_per_family=4,
        train_steps=16,
        lr=0.5,
        split_seed=0,
    )
    assert report["status"] == "v25_01_heldout_eval_ok"
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["n_heldout"] == 16      # 4 families × 4
    assert report["n_train"] == 48        # 4 families × 12
    assert math.isfinite(report["train_loss_final"])
    assert math.isfinite(report["heldout_loss_mean"])
    assert report["heldout_generalizes"] is True
    assert report["train_loss_delta"] >= 0.0


def test_run_heldout_eval_writes_files(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    examples = _make_examples(["a", "b"], 10)
    _write_distill_examples(artifacts_dir / "distill_examples.jsonl", examples)

    out_dir = tmp_path / "out"
    run_v25_01_heldout_eval(
        artifacts_dir, out_dir,
        held_out_per_family=3,
        train_steps=8,
        lr=0.5,
    )
    assert (out_dir / HELDOUT_EVAL_REPORT_FILENAME).exists()
    assert (out_dir / SPLIT_MANIFEST_FILENAME).exists()


def test_run_heldout_eval_per_family_keys(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    examples = _make_examples(["arithmetic_short", "logic_short"], 10)
    _write_distill_examples(artifacts_dir / "distill_examples.jsonl", examples)

    report = run_v25_01_heldout_eval(
        artifacts_dir, tmp_path / "out",
        held_out_per_family=3, train_steps=8, lr=0.5,
    )
    assert set(report["per_family"]) == {"arithmetic_short", "logic_short"}
    for fam_data in report["per_family"].values():
        assert "n" in fam_data
        assert "loss_mean" in fam_data
        assert math.isfinite(fam_data["loss_mean"])


def test_run_heldout_eval_missing_examples_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_v25_01_heldout_eval(tmp_path / "nonexistent", tmp_path / "out")


def test_run_heldout_eval_split_manifest(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    examples = _make_examples(["x", "y"], 8)
    _write_distill_examples(artifacts_dir / "distill_examples.jsonl", examples)

    out_dir = tmp_path / "out"
    run_v25_01_heldout_eval(artifacts_dir, out_dir, held_out_per_family=2, train_steps=4, lr=0.5)
    manifest = json.loads((out_dir / SPLIT_MANIFEST_FILENAME).read_text())
    assert manifest["n_total"] == 16
    assert manifest["n_heldout"] == 4
    assert manifest["n_train"] == 12
    assert len(manifest["heldout_sample_ids"]) == 4
