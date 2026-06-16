from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from src.v26_graph_decoder import (
    BOS_ID,
    CHECKPOINT_FILENAME,
    EOS_ID,
    PAD_ID,
    SCHEMA_VERSION,
    FullGraphDecoderConfig,
    FullGraphDecoderLM,
    GraphTokenizer,
    generate,
    load_adjacency_metadata,
    load_checkpoint,
    save_checkpoint,
    train_full_graph_decoder_model,
)


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


def _make_examples(n: int = 8) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append({
            "sample_id": f"arith_{i}",
            "input": f"Compute {i} + {i + 1}.",
            "target": f"reasoning: Add the numbers.\nanswer: {i + i + 1}",
            "split": "train",
        })
    return rows


def _make_distill_file(path: Path, n: int = 8) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in _make_examples(n):
            fh.write(json.dumps(ex) + "\n")
    return path


def _make_tiny_model(tmp_path: Path, block_size: int = 16) -> tuple[FullGraphDecoderLM, GraphTokenizer]:
    examples = _make_examples(4)
    tokenizer = GraphTokenizer.build(examples)
    adj_path = _make_adjacency_json(tmp_path / "adj.json")
    adjacency_name, n_graph_nodes, graph_bias = load_adjacency_metadata(adj_path, block_size)
    config = FullGraphDecoderConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        hidden_dim=16,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        graph_bias_weight=0.1,
        n_graph_nodes=n_graph_nodes,
        adjacency_name=adjacency_name,
    )
    return FullGraphDecoderLM(config, graph_bias=graph_bias), tokenizer


def test_tokenizer_roundtrip():
    tok = GraphTokenizer.build(_make_examples(3))
    ids = tok.encode("Compute 2 + 3.", add_bos=True, add_eos=True)
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID
    assert tok.token_to_id["<PAD>"] == PAD_ID
    assert tok.decode(ids) == "Compute 2 + 3."


def test_model_forward_shape(tmp_path):
    model, tokenizer = _make_tiny_model(tmp_path)
    idx = torch.tensor([[BOS_ID, tokenizer.token_to_id["Compute"], tokenizer.token_to_id["0"]]], dtype=torch.long)
    logits, loss = model(idx, idx)
    assert logits.shape == (1, 3, tokenizer.vocab_size)
    assert loss is not None
    assert math.isfinite(loss.item())


def test_generation_schema_fields(tmp_path):
    model, tokenizer = _make_tiny_model(tmp_path)
    out = generate(model, tokenizer, "Compute 1 + 2.", max_tokens=5, seed=123)
    assert out["schema_version"] == SCHEMA_VERSION
    assert out["status"] == "generated_autoregressive"
    assert out["adjacency_name"] == "qwen_topk_k2"
    assert out["prompt"] == "Compute 1 + 2."
    assert isinstance(out["text"], str)
    assert out["token_count"] <= 5
    assert out["teacher_checkpoint_loaded"] is False
    assert out["raw_weight_payload_in_graph"] is False
    assert out["bounded_active_adjacency"] is True


def test_checkpoint_save_load_roundtrip(tmp_path):
    model, tokenizer = _make_tiny_model(tmp_path)
    with torch.no_grad():
        model.W_out.weight.fill_(0.25)
    ckpt = save_checkpoint(model, tokenizer, tmp_path / "ckpt", train_report={"ok": True})
    model2, tok2 = load_checkpoint(ckpt)
    assert tok2.token_to_id == tokenizer.token_to_id
    assert model2.config.schema_version == SCHEMA_VERSION
    assert model2.config.block_size == model.config.block_size
    assert abs(model2.W_out.weight.mean().item() - 0.25) < 1e-6
    data = json.loads(ckpt.read_text(encoding="utf-8"))
    assert data["train_report"]["ok"] is True
    assert data["teacher_checkpoint_loaded"] is False
    assert data["raw_weight_payload_in_graph"] is False
    assert data["bounded_active_adjacency"] is True


def test_tiny_training_run_produces_finite_loss(tmp_path):
    adj_path = _make_adjacency_json(tmp_path / "adj.json")
    _make_distill_file(tmp_path / "teacher" / "distill_examples.jsonl", n=6)
    result = train_full_graph_decoder_model(
        adj_path,
        tmp_path / "teacher",
        tmp_path / "out",
        block_size=32,
        hidden_dim=16,
        n_layers=1,
        n_heads=4,
        epochs=3,
        lr=5e-3,
        batch_size=2,
    )
    report = result["report"]
    assert Path(result["checkpoint"]).exists()
    assert math.isfinite(report["loss_initial"])
    assert math.isfinite(report["loss_final"])
    assert len(report["loss_curve"]) >= 1
    assert report["n_examples"] == 6


def test_cli_train_smoke(tmp_path):
    adj_path = _make_adjacency_json(tmp_path / "adj.json")
    _make_distill_file(tmp_path / "teacher" / "distill_examples.jsonl", n=4)
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_v26_graph_decoder_train.py",
            "--adjacency-path",
            str(adj_path),
            "--teacher-artifacts",
            str(tmp_path / "teacher"),
            "--output-dir",
            str(tmp_path / "out"),
            "--block-size",
            "32",
            "--hidden-dim",
            "16",
            "--n-layers",
            "1",
            "--n-heads",
            "4",
            "--epochs",
            "2",
            "--batch-size",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "checkpoint:" in proc.stdout
    assert (tmp_path / "out" / CHECKPOINT_FILENAME).exists()


def test_cli_generate_smoke(tmp_path):
    adj_path = _make_adjacency_json(tmp_path / "adj.json")
    _make_distill_file(tmp_path / "teacher" / "distill_examples.jsonl", n=4)
    result = train_full_graph_decoder_model(
        adj_path,
        tmp_path / "teacher",
        tmp_path / "out",
        block_size=32,
        hidden_dim=16,
        n_layers=1,
        n_heads=4,
        epochs=2,
        batch_size=2,
    )
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_v26_graph_decoder_generate.py",
            "--checkpoint",
            result["checkpoint"],
            "--prompt",
            "Compute 1 + 2.",
            "--max-tokens",
            "4",
            "--seed",
            "7",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["token_count"] <= 4


def test_block_size_truncation_generation(tmp_path):
    model, tokenizer = _make_tiny_model(tmp_path, block_size=4)
    prompt = "Compute 0 + 1. Compute 1 + 2. Compute 2 + 3."
    out = generate(model, tokenizer, prompt, max_tokens=3, seed=1)
    assert out["token_count"] <= 3


def test_parameter_count_nontrivial(tmp_path):
    examples = _make_examples(4)
    tokenizer = GraphTokenizer.build(examples)
    adj_path = _make_adjacency_json(tmp_path / "adj.json")
    adjacency_name, n_graph_nodes, graph_bias = load_adjacency_metadata(adj_path, 32)
    config = FullGraphDecoderConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=32,
        hidden_dim=32,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        graph_bias_weight=0.0,
        n_graph_nodes=n_graph_nodes,
        adjacency_name=adjacency_name,
    )
    model = FullGraphDecoderLM(config, graph_bias=graph_bias)
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 10_000
