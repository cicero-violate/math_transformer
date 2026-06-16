"""Graph-autoregressive language model over the qwen_topk_k2 sparse topology.

Architecture:
  - Token embedding matrix (vocab_size × hidden_dim)
  - Stateful graph nodes (n_nodes × hidden_dim), reset per example
  - Message-passing through the fixed k=2 adjacency (n_steps per token)
  - Mean-pool of node features → linear projection → next-token logits
  - Autoregressive generation with temperature + top-p sampling
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

SCHEMA_VERSION = "v26_graph_ar.v1"
CHECKPOINT_FILENAME = "v26_graph_ar_checkpoint.json"
TRAIN_REPORT_FILENAME = "v26_graph_ar_train_report.json"

TOKEN_RE = re.compile(r"[A-Za-z_]+|\d+|[^\w\s]", re.UNICODE)
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _join_tokens(tokens: Iterable[str]) -> str:
    text = ""
    for tok in tokens:
        if not text:
            text = tok
        elif re.match(r"^[.,:;!?)]$", tok):
            text += tok
        else:
            text += " " + tok
    return text


class GraphTokenizer:
    """Vocabulary built from training examples; encode/decode token sequences."""

    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {v: k for k, v in token_to_id.items()}

    @classmethod
    def build(cls, examples: list[dict[str, Any]]) -> "GraphTokenizer":
        counts: Counter[str] = Counter()
        for ex in examples:
            counts.update(_tokenize(str(ex.get("input") or ex.get("input_text") or ex.get("prompt") or "")))
            counts.update(_tokenize(str(ex.get("target") or ex.get("response") or "")))
        token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for token, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.token_to_id.get(tok, UNK_ID) for tok in _tokenize(text)]
        if add_bos:
            ids = [BOS_ID] + ids
        if add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: list[int]) -> str:
        tokens: list[str] = []
        for id_ in ids:
            tok = self.id_to_token.get(id_, "<UNK>")
            if tok in ("<PAD>", "<BOS>"):
                continue
            if tok == "<EOS>":
                break
            tokens.append(tok)
        return _join_tokens(tokens)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"schema_version": "v26_graph_tokenizer.v1", "token_to_id": self.token_to_id}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "GraphTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({str(k): int(v) for k, v in data["token_to_id"].items()})


# ---------------------------------------------------------------------------
# Adjacency loading
# ---------------------------------------------------------------------------

def load_adjacency_for_model(
    adjacency_path: str | Path,
) -> tuple[torch.Tensor, torch.Tensor, int, list[str]]:
    """Return (src_idx, dst_idx, n_nodes, sorted_node_ids) from a selected_adjacency JSON."""
    data = json.loads(Path(adjacency_path).read_text(encoding="utf-8"))
    edges = data["edges"]
    all_node_ids: list[str] = sorted(
        {str(e["src_id"]) for e in edges} | {str(e["dst_id"]) for e in edges}
    )
    node_to_idx = {nid: i for i, nid in enumerate(all_node_ids)}
    src_idx = torch.tensor([node_to_idx[str(e["src_id"])] for e in edges], dtype=torch.long)
    dst_idx = torch.tensor([node_to_idx[str(e["dst_id"])] for e in edges], dtype=torch.long)
    return src_idx, dst_idx, len(all_node_ids), all_node_ids


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GraphARModel(nn.Module):
    """Graph-autoregressive LM: stateful message-passing over a fixed sparse adjacency."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        n_nodes: int,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        *,
        n_steps: int = 2,
        adjacency_name: str = "qwen_topk_k2",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_nodes = n_nodes
        self.n_steps = n_steps
        self.adjacency_name = adjacency_name
        # Token embedding
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD_ID)
        # Message-passing weights
        self.W_msg = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_self = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        # Output projection
        self.W_out = nn.Linear(hidden_dim, vocab_size)
        # Fixed topology (not trainable)
        self.register_buffer("src_idx", src_idx)
        self.register_buffer("dst_idx", dst_idx)

    def zero_state(self, device: str | torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(self.n_nodes, self.hidden_dim, device=device)

    def encode_tokens(self, token_ids: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Update node state h given a batch of new token ids (1-D long tensor)."""
        emb = self.embed(token_ids)          # (seq, hidden)
        context = emb.mean(0)               # (hidden,)  — mean-pool over sequence
        h = h + context.unsqueeze(0)        # broadcast to all nodes
        for _ in range(self.n_steps):
            msgs = self.W_msg(h[self.src_idx])          # (n_edges, hidden)
            agg = torch.zeros_like(h)
            agg.index_add_(0, self.dst_idx, msgs)        # aggregate at dst nodes
            h = F.relu(self.norm(self.W_self(h) + agg))
        return h

    def decode_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Next-token logits from current node state."""
        return self.W_out(h.mean(0))   # (vocab_size,)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def _tensor_to_list(t: torch.Tensor) -> list:
    return t.detach().cpu().tolist()


def save_checkpoint(
    model: GraphARModel,
    tokenizer: GraphTokenizer,
    output_dir: str | Path,
    *,
    node_ids: list[str] | None = None,
    train_report: dict[str, Any] | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "adjacency_name": model.adjacency_name,
        "vocab_size": model.vocab_size,
        "hidden_dim": model.hidden_dim,
        "n_nodes": model.n_nodes,
        "n_steps": model.n_steps,
        "teacher_checkpoint_loaded": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
        "node_ids": node_ids or [],
        "token_to_id": tokenizer.token_to_id,
        "state_dict": {k: _tensor_to_list(v) for k, v in model.state_dict().items()},
    }
    if train_report:
        state["train_report"] = train_report
    ckpt_path = out / CHECKPOINT_FILENAME
    ckpt_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return ckpt_path


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[GraphARModel, GraphTokenizer]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tokenizer = GraphTokenizer({str(k): int(v) for k, v in data["token_to_id"].items()})
    src_list: list[int] = data["state_dict"]["src_idx"]
    dst_list: list[int] = data["state_dict"]["dst_idx"]
    src_idx = torch.tensor(src_list, dtype=torch.long)
    dst_idx = torch.tensor(dst_list, dtype=torch.long)
    model = GraphARModel(
        vocab_size=int(data["vocab_size"]),
        hidden_dim=int(data["hidden_dim"]),
        n_nodes=int(data["n_nodes"]),
        src_idx=src_idx,
        dst_idx=dst_idx,
        n_steps=int(data["n_steps"]),
        adjacency_name=str(data["adjacency_name"]),
    )
    sd = {k: torch.tensor(v) for k, v in data["state_dict"].items()}
    model.load_state_dict(sd)
    model = model.to(device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _load_examples(teacher_artifacts: str | Path) -> list[dict[str, Any]]:
    path = Path(teacher_artifacts) / "distill_examples.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def train_graph_ar_model(
    adjacency_path: str | Path,
    teacher_artifacts: str | Path,
    output_dir: str | Path,
    *,
    hidden_dim: int = 64,
    n_steps: int = 2,
    epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train a GraphARModel on distill_examples.jsonl; save checkpoint and report."""
    examples = _load_examples(teacher_artifacts)
    tokenizer = GraphTokenizer.build(examples)
    src_idx, dst_idx, n_nodes, node_ids = load_adjacency_for_model(adjacency_path)
    model = GraphARModel(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=hidden_dim,
        n_nodes=n_nodes,
        src_idx=src_idx,
        dst_idx=dst_idx,
        n_steps=n_steps,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Pre-encode all examples
    encoded: list[tuple[list[int], list[int]]] = []
    for ex in examples:
        input_text = str(ex.get("input") or ex.get("input_text") or ex.get("prompt") or "")
        target_text = str(ex.get("target") or ex.get("response") or "")
        prompt_ids = tokenizer.encode(input_text)
        target_ids = tokenizer.encode(target_text, add_bos=True, add_eos=True)
        if prompt_ids and len(target_ids) >= 2:
            encoded.append((prompt_ids, target_ids))

    if not encoded:
        raise ValueError("no usable training examples after encoding")

    losses: list[float] = []
    log_interval = max(1, epochs // 10)
    for epoch in range(epochs):
        epoch_loss = 0.0
        for prompt_ids, target_ids in encoded:
            optimizer.zero_grad()
            h = model.zero_state(device=device)
            # Encode prompt into node state
            prompt_t = torch.tensor(prompt_ids, dtype=torch.long, device=device)
            h = model.encode_tokens(prompt_t, h)
            # Teacher-forced generation over target sequence
            loss = torch.zeros(1, device=device)
            for i in range(len(target_ids) - 1):
                logits = model.decode_logits(h)
                target_t = torch.tensor([target_ids[i + 1]], dtype=torch.long, device=device)
                loss = loss + F.cross_entropy(logits.unsqueeze(0), target_t)
                tok_t = torch.tensor([target_ids[i]], dtype=torch.long, device=device)
                h = model.encode_tokens(tok_t, h)
            n_steps_loss = len(target_ids) - 1
            loss = loss / n_steps_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        mean_epoch_loss = epoch_loss / len(encoded)
        losses.append(mean_epoch_loss)
        if epoch % log_interval == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:4d}/{epochs}  loss={mean_epoch_loss:.4f}")

    out = Path(output_dir)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "n_examples": len(encoded),
        "vocab_size": tokenizer.vocab_size,
        "hidden_dim": hidden_dim,
        "n_nodes": n_nodes,
        "n_steps": n_steps,
        "epochs": epochs,
        "lr": lr,
        "loss_initial": losses[0],
        "loss_final": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "loss_curve": losses[::max(1, epochs // 20)],  # subsample for brevity
        "teacher_checkpoint_loaded": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
    }
    (out / TRAIN_REPORT_FILENAME).parent.mkdir(parents=True, exist_ok=True)
    (out / TRAIN_REPORT_FILENAME).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    ckpt_path = save_checkpoint(model, tokenizer, out, node_ids=node_ids, train_report=report)
    return {"checkpoint": str(ckpt_path), "report": report}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    model: GraphARModel,
    tokenizer: GraphTokenizer,
    prompt: str,
    *,
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Autoregressive generation conditioned on prompt via graph state."""
    model.eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        h = model.zero_state(device=device)
        prompt_ids = tokenizer.encode(prompt)
        if prompt_ids:
            h = model.encode_tokens(torch.tensor(prompt_ids, dtype=torch.long, device=device), h)
        # Start with BOS
        generated_ids: list[int] = []
        current_id = BOS_ID
        for _ in range(max_tokens):
            h = model.encode_tokens(torch.tensor([current_id], dtype=torch.long, device=device), h)
            logits = model.decode_logits(h)
            # Temperature
            logits = logits / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            # Top-p nucleus filtering
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=0)
            keep = torch.cat([torch.tensor([True], device=device), (cumsum[:-1] < top_p)])
            sorted_probs = sorted_probs * keep.float()
            sorted_probs = sorted_probs / sorted_probs.sum().clamp(min=1e-9)
            sampled_pos = torch.multinomial(sorted_probs, 1, generator=rng).item()
            next_id = sorted_indices[sampled_pos].item()
            if next_id == EOS_ID:
                break
            generated_ids.append(int(next_id))
            current_id = int(next_id)
    text = tokenizer.decode(generated_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "generated_autoregressive",
        "adjacency_name": model.adjacency_name,
        "prompt": prompt,
        "text": text,
        "token_count": len(generated_ids),
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "teacher_checkpoint_loaded": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
    }
