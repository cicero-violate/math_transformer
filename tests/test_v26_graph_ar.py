from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from src.v26_graph_ar import (
    SCHEMA_VERSION,
    CHECKPOINT_FILENAME,
    BOS_ID, EOS_ID, UNK_ID, PAD_ID,
    GraphTokenizer,
    GraphARModel,
    load_adjacency_for_model,
    save_checkpoint,
    load_checkpoint,
    train_graph_ar_model,
    generate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAMILIES = ["arithmetic_short", "logic_short", "symbolic_short", "project_specific"]


def _make_adjacency_json(path: Path, n_src: int = 4, k: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    edges = []
    idx = 0
    for src_i in range(n_src):
        for d in range(1, k + 1):
            dst_i = (src_i + d) % (n_src + k)
            edges.append({
                "edge_id": f"e{idx:04d}",
                "src_id": f"node_{src_i:02d}",
                "dst_id": f"node_{dst_i:02d}",
                "weight": 0.9 - idx * 0.01,
                "relation": "qk_affinity_prior",
                "score_name": "normalized_frobenius",
                "source": "G_0",
                "metadata": {},
            })
            idx += 1
    all_nodes = sorted({str(e["src_id"]) for e in edges} | {str(e["dst_id"]) for e in edges})
    data = {
        "schema_version": "qwen_selected_adjacency.v1",
        "adjacency_name": "qwen_topk_k2",
        "k": k,
        "bounded": True,
        "source": "G_0",
        "selection_policy": "per_source_topk_score_desc",
        "node_count": len(all_nodes),
        "edge_count": len(edges),
        "edge_score_name": "normalized_frobenius",
        "edges": edges,
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def _make_examples(n_per_family: int = 8) -> list[dict]:
    rows = []
    for fam in FAMILIES:
        for i in range(n_per_family):
            rows.append({
                "sample_id": f"{fam}_{i}",
                "family": fam,
                "input": f"Compute {i} + {i + 1}.",
                "target": f"reasoning: Add the numbers.\nanswer: {i + i + 1}",
                "split": "train",
            })
    return rows


def _make_distill_file(path: Path, n_per_family: int = 8) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in _make_examples(n_per_family):
            fh.write(json.dumps(ex) + "\n")
    return path


def _make_model(tmp_path: Path, vocab_size: int = 32, hidden_dim: int = 8, n_src: int = 4):
    adj_path = _make_adjacency_json(tmp_path / "adj.json", n_src=n_src)
    src_idx, dst_idx, n_nodes, node_ids = load_adjacency_for_model(adj_path)
    model = GraphARModel(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        n_nodes=n_nodes,
        src_idx=src_idx,
        dst_idx=dst_idx,
        n_steps=1,
    )
    return model, node_ids


# ---------------------------------------------------------------------------
# GraphTokenizer
# ---------------------------------------------------------------------------

def test_tokenizer_build(tmp_path):
    examples = _make_examples(4)
    tok = GraphTokenizer.build(examples)
    assert tok.vocab_size >= len(["<PAD>", "<BOS>", "<EOS>", "<UNK>"]) + 4
    assert "<PAD>" in tok.token_to_id
    assert tok.token_to_id["<PAD>"] == PAD_ID
    assert tok.token_to_id["<BOS>"] == BOS_ID
    assert tok.token_to_id["<EOS>"] == EOS_ID
    assert tok.token_to_id["<UNK>"] == UNK_ID


def test_tokenizer_encode_decode(tmp_path):
    examples = _make_examples(4)
    tok = GraphTokenizer.build(examples)
    text = "Add the numbers."
    ids = tok.encode(text)
    assert all(isinstance(i, int) for i in ids)
    # Known tokens should not produce UNK
    for token in ["Add", "the", "numbers"]:
        assert tok.token_to_id.get(token, UNK_ID) != UNK_ID

    decoded = tok.decode(ids)
    assert isinstance(decoded, str)
    assert len(decoded) > 0


def test_tokenizer_bos_eos(tmp_path):
    tok = GraphTokenizer.build(_make_examples(2))
    ids = tok.encode("Compute 2 + 3.", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID


def test_tokenizer_save_load(tmp_path):
    tok = GraphTokenizer.build(_make_examples(2))
    path = tmp_path / "vocab.json"
    tok.save(path)
    tok2 = GraphTokenizer.load(path)
    assert tok2.token_to_id == tok.token_to_id
    assert tok2.vocab_size == tok.vocab_size


def test_tokenizer_unknown_token():
    tok = GraphTokenizer.build(_make_examples(1))
    ids = tok.encode("xyzzy_nonsense_8675309")
    # All tokens are UNK
    assert all(i == UNK_ID for i in ids)


# ---------------------------------------------------------------------------
# load_adjacency_for_model
# ---------------------------------------------------------------------------

def test_load_adjacency_for_model(tmp_path):
    adj_path = _make_adjacency_json(tmp_path / "adj.json", n_src=4, k=2)
    src_idx, dst_idx, n_nodes, node_ids = load_adjacency_for_model(adj_path)
    assert src_idx.dtype == torch.long
    assert dst_idx.dtype == torch.long
    assert src_idx.shape == dst_idx.shape
    assert n_nodes >= 4
    assert len(node_ids) == n_nodes
    assert src_idx.max().item() < n_nodes
    assert dst_idx.max().item() < n_nodes


# ---------------------------------------------------------------------------
# GraphARModel
# ---------------------------------------------------------------------------

def test_model_zero_state(tmp_path):
    model, _ = _make_model(tmp_path)
    h = model.zero_state()
    assert h.shape == (model.n_nodes, model.hidden_dim)
    assert h.sum().item() == 0.0


def test_model_encode_tokens(tmp_path):
    model, _ = _make_model(tmp_path, vocab_size=32, hidden_dim=8)
    h = model.zero_state()
    ids = torch.tensor([1, 2, 3], dtype=torch.long)
    h2 = model.encode_tokens(ids, h)
    assert h2.shape == h.shape
    # State changed from zero
    assert h2.abs().sum().item() > 0.0


def test_model_decode_logits(tmp_path):
    model, _ = _make_model(tmp_path, vocab_size=32, hidden_dim=8)
    h = model.zero_state()
    ids = torch.tensor([BOS_ID], dtype=torch.long)
    h = model.encode_tokens(ids, h)
    logits = model.decode_logits(h)
    assert logits.shape == (32,)
    assert all(math.isfinite(x) for x in logits.tolist())


def test_model_encode_decode_chain(tmp_path):
    model, _ = _make_model(tmp_path, vocab_size=32, hidden_dim=8)
    h = model.zero_state()
    for tok_id in [BOS_ID, 5, 10, 15]:
        h = model.encode_tokens(torch.tensor([tok_id], dtype=torch.long), h)
    logits = model.decode_logits(h)
    assert logits.shape == (32,)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def test_checkpoint_round_trip(tmp_path):
    examples = _make_examples(2)
    tok = GraphTokenizer.build(examples)
    model, node_ids = _make_model(tmp_path / "model", vocab_size=tok.vocab_size)
    ckpt_path = save_checkpoint(model, tok, tmp_path / "ckpt", node_ids=node_ids)
    assert ckpt_path.exists()
    model2, tok2 = load_checkpoint(ckpt_path)
    assert model2.vocab_size == model.vocab_size
    assert model2.hidden_dim == model.hidden_dim
    assert model2.n_nodes == model.n_nodes
    assert tok2.vocab_size == tok.vocab_size


def test_checkpoint_safety_flags(tmp_path):
    examples = _make_examples(2)
    tok = GraphTokenizer.build(examples)
    model, _ = _make_model(tmp_path / "model", vocab_size=tok.vocab_size)
    ckpt_path = save_checkpoint(model, tok, tmp_path / "ckpt")
    data = json.loads(ckpt_path.read_text(encoding="utf-8"))
    assert data["teacher_checkpoint_loaded"] is False
    assert data["raw_weight_payload_in_graph"] is False
    assert data["bounded_active_adjacency"] is True
    assert data["schema_version"] == SCHEMA_VERSION


def test_checkpoint_state_dict_preserved(tmp_path):
    examples = _make_examples(2)
    tok = GraphTokenizer.build(examples)
    model, _ = _make_model(tmp_path / "model", vocab_size=tok.vocab_size, hidden_dim=8)
    # Perturb weights
    with torch.no_grad():
        model.W_out.weight.fill_(3.14)
    ckpt_path = save_checkpoint(model, tok, tmp_path / "ckpt")
    model2, _ = load_checkpoint(ckpt_path)
    assert abs(model2.W_out.weight.mean().item() - 3.14) < 1e-4


# ---------------------------------------------------------------------------
# train_graph_ar_model
# ---------------------------------------------------------------------------

def test_train_graph_ar_model_basic(tmp_path):
    adj_path = _make_adjacency_json(tmp_path / "adj.json", n_src=4)
    distill_path = _make_distill_file(tmp_path / "teacher" / "distill_examples.jsonl", n_per_family=4)
    result = train_graph_ar_model(
        adj_path,
        tmp_path / "teacher",
        tmp_path / "out",
        hidden_dim=8,
        n_steps=1,
        epochs=5,
        lr=1e-2,
    )
    report = result["report"]
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["n_examples"] > 0
    assert math.isfinite(report["loss_initial"])
    assert math.isfinite(report["loss_final"])
    assert report["teacher_checkpoint_loaded"] is False
    assert report["raw_weight_payload_in_graph"] is False
    assert report["bounded_active_adjacency"] is True


def test_train_graph_ar_model_loss_decreases(tmp_path):
    adj_path = _make_adjacency_json(tmp_path / "adj.json", n_src=4)
    _make_distill_file(tmp_path / "teacher" / "distill_examples.jsonl", n_per_family=8)
    result = train_graph_ar_model(
        adj_path, tmp_path / "teacher", tmp_path / "out",
        hidden_dim=16, n_steps=1, epochs=30, lr=5e-3,
    )
    report = result["report"]
    assert report["loss_final"] < report["loss_initial"], (
        f"loss did not decrease: {report['loss_initial']:.4f} → {report['loss_final']:.4f}"
    )


def test_train_graph_ar_model_writes_checkpoint(tmp_path):
    adj_path = _make_adjacency_json(tmp_path / "adj.json", n_src=3)
    _make_distill_file(tmp_path / "teacher" / "distill_examples.jsonl", n_per_family=4)
    result = train_graph_ar_model(
        adj_path, tmp_path / "teacher", tmp_path / "out",
        hidden_dim=8, n_steps=1, epochs=3,
    )
    assert Path(result["checkpoint"]).exists()
    assert (tmp_path / "out" / CHECKPOINT_FILENAME).exists()


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

def test_generate_basic(tmp_path):
    adj_path = _make_adjacency_json(tmp_path / "adj.json", n_src=4)
    _make_distill_file(tmp_path / "teacher" / "distill_examples.jsonl", n_per_family=4)
    result = train_graph_ar_model(
        adj_path, tmp_path / "teacher", tmp_path / "out",
        hidden_dim=16, n_steps=1, epochs=5,
    )
    model, tokenizer = load_checkpoint(result["checkpoint"])
    out = generate(model, tokenizer, "Compute 3 + 4.", max_tokens=16, seed=0)
    assert out["schema_version"] == SCHEMA_VERSION
    assert out["status"] == "generated_autoregressive"
    assert isinstance(out["text"], str)
    assert out["token_count"] >= 0
    assert out["token_count"] <= 16
    assert out["teacher_checkpoint_loaded"] is False
    assert out["bounded_active_adjacency"] is True


def test_generate_empty_prompt(tmp_path):
    adj_path = _make_adjacency_json(tmp_path / "adj.json", n_src=3)
    _make_distill_file(tmp_path / "teacher" / "distill_examples.jsonl", n_per_family=4)
    result = train_graph_ar_model(
        adj_path, tmp_path / "teacher", tmp_path / "out",
        hidden_dim=8, n_steps=1, epochs=3,
    )
    model, tokenizer = load_checkpoint(result["checkpoint"])
    out = generate(model, tokenizer, "", max_tokens=8)
    assert isinstance(out["text"], str)


def test_generate_deterministic_seed(tmp_path):
    adj_path = _make_adjacency_json(tmp_path / "adj.json", n_src=4)
    _make_distill_file(tmp_path / "teacher" / "distill_examples.jsonl", n_per_family=4)
    result = train_graph_ar_model(
        adj_path, tmp_path / "teacher", tmp_path / "out",
        hidden_dim=8, n_steps=1, epochs=3,
    )
    model, tokenizer = load_checkpoint(result["checkpoint"])
    out1 = generate(model, tokenizer, "What is 2 + 3?", max_tokens=8, seed=42)
    out2 = generate(model, tokenizer, "What is 2 + 3?", max_tokens=8, seed=42)
    assert out1["text"] == out2["text"]
