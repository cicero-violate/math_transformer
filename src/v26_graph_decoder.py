"""Full local graph-aware decoder-only language model.

This v26 decoder keeps the v26 tokenizer/checkpoint ergonomics, but replaces mean-pooled
prompt injection with a true causal sequence model over ordered tokens.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

SCHEMA_VERSION = "v26_graph_decoder.v1"
CHECKPOINT_FILENAME = "v26_graph_decoder_checkpoint.json"
TRAIN_REPORT_FILENAME = "v26_graph_decoder_train_report.json"

TOKEN_RE = re.compile(r"[A-Za-z_]+|\d+|[^\w\s]", re.UNICODE)
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3


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
    """Vocabulary built from distillation examples; compatible with v26."""

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
            tok = self.id_to_token.get(int(id_), "<UNK>")
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
            json.dumps({"schema_version": "v26_graph_decoder_tokenizer.v1", "token_to_id": self.token_to_id}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "GraphTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({str(k): int(v) for k, v in data["token_to_id"].items()})


@dataclass
class FullGraphDecoderConfig:
    vocab_size: int
    block_size: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    dropout: float
    graph_bias_weight: float
    n_graph_nodes: int
    adjacency_name: str
    schema_version: str = SCHEMA_VERSION


def load_adjacency_metadata(
    adjacency_path: str | Path,
    block_size: int,
) -> tuple[str, int, torch.Tensor]:
    """Load bounded topology metadata and build a small optional attention bias."""
    data = json.loads(Path(adjacency_path).read_text(encoding="utf-8"))
    edges = data.get("edges", [])
    all_node_ids = sorted({str(e["src_id"]) for e in edges} | {str(e["dst_id"]) for e in edges})
    node_to_idx = {node_id: i for i, node_id in enumerate(all_node_ids)}
    bias = torch.zeros(block_size, block_size, dtype=torch.float32)
    for edge in edges:
        src = node_to_idx.get(str(edge.get("src_id")))
        dst = node_to_idx.get(str(edge.get("dst_id")))
        if src is None or dst is None:
            continue
        i = dst % block_size
        j = src % block_size
        if j <= i:
            bias[i, j] += float(edge.get("weight", 1.0))
    if bias.abs().max().item() > 0:
        bias = bias / bias.abs().max().clamp(min=1e-9)
    return str(data.get("adjacency_name", "qwen_topk_k2")), len(all_node_ids), bias


class CausalSelfAttention(nn.Module):
    def __init__(self, config: FullGraphDecoderConfig, graph_bias: torch.Tensor | None = None) -> None:
        super().__init__()
        if config.hidden_dim % config.n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_dim // config.n_heads
        self.graph_bias_weight = float(config.graph_bias_weight)
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        causal = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal.view(1, 1, config.block_size, config.block_size))
        if graph_bias is None:
            graph_bias = torch.zeros(config.block_size, config.block_size, dtype=torch.float32)
        self.register_buffer("graph_bias", graph_bias.float().view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(~self.causal_mask[:, :, :seq_len, :seq_len], float("-inf"))
        if self.graph_bias_weight:
            att = att + self.graph_bias_weight * self.graph_bias[:, :, :seq_len, :seq_len]
        probs = F.softmax(att, dim=-1)
        probs = self.attn_dropout(probs)
        y = probs @ v
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, config: FullGraphDecoderConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, config: FullGraphDecoderConfig, graph_bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_dim)
        self.attn = CausalSelfAttention(config, graph_bias=graph_bias)
        self.ln_2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class FullGraphDecoderLM(nn.Module):
    """Decoder-only LM: ordered context x_1:t -> p(x_t+1)."""

    def __init__(self, config: FullGraphDecoderConfig, graph_bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_dim, padding_idx=PAD_ID)
        self.pos_emb = nn.Embedding(config.block_size, config.hidden_dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config, graph_bias=graph_bias) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.W_out = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    @property
    def adjacency_name(self) -> str:
        return self.config.adjacency_name

    @property
    def block_size(self) -> int:
        return self.config.block_size

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        _batch, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.config.block_size}")
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.W_out(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss


def _tensor_to_list(t: torch.Tensor) -> list:
    return t.detach().cpu().tolist()


def _state_dict_from_json(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(v) for k, v in payload.items()}


def save_checkpoint(
    model: FullGraphDecoderLM,
    tokenizer: GraphTokenizer,
    output_dir: str | Path,
    *,
    train_report: dict[str, Any] | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(model.config),
        "token_to_id": tokenizer.token_to_id,
        "state_dict": {k: _tensor_to_list(v) for k, v in model.state_dict().items()},
        "teacher_checkpoint_loaded": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
    }
    if train_report is not None:
        payload["train_report"] = train_report
    ckpt_path = out / CHECKPOINT_FILENAME
    ckpt_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return ckpt_path


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[FullGraphDecoderLM, GraphTokenizer]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tokenizer = GraphTokenizer({str(k): int(v) for k, v in data["token_to_id"].items()})
    config = FullGraphDecoderConfig(**data["config"])
    model = FullGraphDecoderLM(config)
    model.load_state_dict(_state_dict_from_json(data["state_dict"]))
    return model.to(device), tokenizer


def _load_examples(teacher_artifacts: str | Path) -> list[dict[str, Any]]:
    path = Path(teacher_artifacts) / "distill_examples.jsonl"
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _make_sequence(tokenizer: GraphTokenizer, ex: dict[str, Any], block_size: int) -> list[int]:
    prompt = str(ex.get("input") or ex.get("input_text") or ex.get("prompt") or "")
    target = str(ex.get("target") or ex.get("response") or "")
    seq = [BOS_ID] + tokenizer.encode(prompt) + tokenizer.encode(target) + [EOS_ID]
    if len(seq) > block_size + 1:
        seq = seq[-(block_size + 1):]
    return seq


def _make_batch(seqs: list[list[int]], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(seq) - 1 for seq in seqs)
    xs: list[list[int]] = []
    ys: list[list[int]] = []
    for seq in seqs:
        x = seq[:-1]
        y = seq[1:]
        pad = max_len - len(x)
        xs.append(x + [PAD_ID] * pad)
        ys.append(y + [-100] * pad)
    return torch.tensor(xs, dtype=torch.long, device=device), torch.tensor(ys, dtype=torch.long, device=device)


def _parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_full_graph_decoder_model(
    adjacency_path: str | Path,
    teacher_artifacts: str | Path,
    output_dir: str | Path,
    *,
    block_size: int = 128,
    hidden_dim: int = 128,
    n_layers: int = 4,
    n_heads: int = 4,
    dropout: float = 0.0,
    graph_bias_weight: float = 0.0,
    epochs: int = 100,
    lr: float = 2e-3,
    batch_size: int = 8,
    device: str = "cpu",
) -> dict[str, Any]:
    examples = _load_examples(teacher_artifacts)
    tokenizer = GraphTokenizer.build(examples)
    adjacency_name, n_graph_nodes, graph_bias = load_adjacency_metadata(adjacency_path, block_size)
    config = FullGraphDecoderConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        graph_bias_weight=graph_bias_weight,
        n_graph_nodes=n_graph_nodes,
        adjacency_name=adjacency_name,
    )
    torch.manual_seed(0)
    model = FullGraphDecoderLM(config, graph_bias=graph_bias).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    encoded = [_make_sequence(tokenizer, ex, block_size) for ex in examples]
    encoded = [seq for seq in encoded if len(seq) >= 2]
    if not encoded:
        raise ValueError("no usable training examples after encoding")

    losses: list[float] = []
    model.train()
    log_interval = max(1, epochs // 10)
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(encoded), batch_size):
            batch = encoded[start:start + batch_size]
            x, y = _make_batch(batch, device)
            optimizer.zero_grad()
            _logits, loss = model(x, y)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        mean_epoch_loss = epoch_loss / max(1, n_batches)
        losses.append(mean_epoch_loss)
        if epoch % log_interval == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:4d}/{epochs}  loss={mean_epoch_loss:.4f}")

    out = Path(output_dir)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "n_examples": len(encoded),
        "vocab_size": tokenizer.vocab_size,
        "block_size": block_size,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "dropout": dropout,
        "graph_bias_weight": graph_bias_weight,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "loss_initial": losses[0],
        "loss_final": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "loss_curve": losses[::max(1, epochs // 20)],
        "parameter_count": _parameter_count(model),
        "teacher_checkpoint_loaded": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / TRAIN_REPORT_FILENAME).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    ckpt_path = save_checkpoint(model, tokenizer, out, train_report=report)
    return {"checkpoint": str(ckpt_path), "report": report}


def _sample_top_p(logits: torch.Tensor, temperature: float, top_p: float, rng: torch.Generator) -> int:
    logits = logits / max(float(temperature), 1e-6)
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=0)
    keep = torch.cat([torch.tensor([True], device=logits.device), cumsum[:-1] < float(top_p)])
    sorted_probs = sorted_probs * keep.float()
    sorted_probs = sorted_probs / sorted_probs.sum().clamp(min=1e-9)
    sampled_pos = torch.multinomial(sorted_probs, 1, generator=rng).item()
    return int(sorted_indices[sampled_pos].item())


def generate(
    model: FullGraphDecoderLM,
    tokenizer: GraphTokenizer,
    prompt: str,
    *,
    max_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    model = model.to(device)
    model.eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    context = [BOS_ID] + tokenizer.encode(prompt)
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(max_tokens):
            idx = torch.tensor([context[-model.block_size:]], dtype=torch.long, device=device)
            logits, _loss = model(idx)
            next_id = _sample_top_p(logits[0, -1], temperature, top_p, rng)
            if next_id == EOS_ID:
                break
            generated.append(next_id)
            context.append(next_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "generated_autoregressive",
        "adjacency_name": model.adjacency_name,
        "prompt": prompt,
        "text": tokenizer.decode(generated),
        "token_count": len(generated),
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "teacher_checkpoint_loaded": False,
        "raw_weight_payload_in_graph": False,
        "bounded_active_adjacency": True,
    }
