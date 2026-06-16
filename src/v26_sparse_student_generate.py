from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = "v26_sparse_student_generate.v1"
TOKEN_RE = re.compile(r"[A-Za-z_]+|\d+|[^\w\s]", re.UNICODE)


def simple_tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"expected object in {path}:{line_no}")
            yield row


def _finite_float(value: Any, *, name: str) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {out!r}")
    return out


def load_student_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = read_json(path)
    vocab = checkpoint.get("vocab")
    logits = checkpoint.get("token_logits")
    if not isinstance(vocab, list) or not vocab:
        raise ValueError("student checkpoint must contain non-empty vocab list")
    if not isinstance(logits, dict) or not logits:
        raise ValueError("student checkpoint must contain token_logits object")
    clean_vocab = [str(token) for token in vocab]
    clean_logits: dict[str, float] = {}
    for token in clean_vocab:
        if token not in logits:
            raise ValueError(f"token_logits missing vocab token {token!r}")
        clean_logits[token] = _finite_float(logits[token], name=f"token_logits[{token!r}]")
    return {
        "vocab": clean_vocab,
        "token_logits": clean_logits,
        "adjacency_checksum": checkpoint.get("adjacency_checksum"),
    }


def softmax(logits: dict[str, float], *, temperature: float = 1.0) -> dict[str, float]:
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    if not logits:
        raise ValueError("logits must be non-empty")
    scaled = {token: value / temperature for token, value in logits.items()}
    max_logit = max(scaled.values())
    exp_values = {token: math.exp(value - max_logit) for token, value in scaled.items()}
    denom = sum(exp_values.values())
    if denom <= 0 or not math.isfinite(denom):
        raise ValueError("softmax denominator is invalid")
    return {token: value / denom for token, value in exp_values.items()}


def _top_p_filter(probs: dict[str, float], top_p: float) -> dict[str, float]:
    if not (0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")
    rows = sorted(probs.items(), key=lambda item: item[1], reverse=True)
    kept: dict[str, float] = {}
    total = 0.0
    for token, prob in rows:
        kept[token] = prob
        total += prob
        if total >= top_p:
            break
    denom = sum(kept.values())
    return {token: prob / denom for token, prob in kept.items()}


def _sample_token(probs: dict[str, float], rng: random.Random) -> str:
    threshold = rng.random()
    running = 0.0
    last = None
    for token, prob in probs.items():
        running += prob
        last = token
        if running >= threshold:
            return token
    if last is None:
        raise ValueError("cannot sample empty distribution")
    return last


def join_tokens(tokens: Sequence[str]) -> str:
    text = ""
    for token in tokens:
        if not text:
            text = token
        elif re.match(r"^[.,:;!?)]$", token):
            text += token
        elif token == "_":
            text += token
        elif text.endswith("_"):
            text += token
        else:
            text += " " + token
    return text


def generate_from_token_bias(
    checkpoint: dict[str, Any],
    *,
    prompt: str,
    max_tokens: int = 32,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 0,
    prompt_bonus: float = 0.0,
    greedy: bool = False,
) -> dict[str, Any]:
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    logits = dict(checkpoint["token_logits"])
    prompt_counts = Counter(simple_tokenize(prompt))
    if prompt_bonus:
        for token, count in prompt_counts.items():
            if token in logits:
                logits[token] += prompt_bonus * math.log1p(count)
    probs = _top_p_filter(softmax(logits, temperature=temperature), top_p)
    rng = random.Random(seed)
    if greedy:
        ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        generated = [ordered[i % len(ordered)][0] for i in range(max_tokens)]
    else:
        generated = [_sample_token(probs, rng) for _ in range(max_tokens)]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "generated_from_token_bias",
        "mode": "token_bias",
        "prompt": prompt,
        "text": join_tokens(generated),
        "tokens": generated,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "prompt_bonus": prompt_bonus,
        "greedy": greedy,
        "capability_note": "This checkpoint is a trained token-bias distribution, not a prompt-conditioned autoregressive graph transformer.",
    }


def load_distill_examples(teacher_artifacts: str | Path) -> list[dict[str, Any]]:
    path = Path(teacher_artifacts) / "distill_examples.jsonl"
    if not path.exists():
        raise FileNotFoundError(str(path))
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"no examples in {path}")
    return rows


def _jaccard_score(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def retrieve_nearest_response(*, prompt: str, teacher_artifacts: str | Path, top_k: int = 3) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    prompt_tokens = set(simple_tokenize(prompt.lower()))
    scored: list[tuple[float, dict[str, Any]]] = []
    for row in load_distill_examples(teacher_artifacts):
        input_text = str(row.get("input_text") or row.get("input") or row.get("prompt") or "")
        row_tokens = set(simple_tokenize(input_text.lower()))
        scored.append((_jaccard_score(prompt_tokens, row_tokens), row))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    neighbors = [
        {
            "score": score,
            "sample_id": row.get("sample_id"),
            "family": row.get("family"),
            "input_text": row.get("input_text") or row.get("input") or row.get("prompt"),
        }
        for score, row in scored[:top_k]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "retrieved_nearest_teacher_response",
        "mode": "nearest",
        "prompt": prompt,
        "text": str(best.get("target_text") or best.get("target") or best.get("response") or ""),
        "nearest_score": best_score,
        "nearest_sample_id": best.get("sample_id"),
        "nearest_family": best.get("family"),
        "neighbors": neighbors,
        "capability_note": "Nearest mode is artifact-backed retrieval over distillation examples; it is useful for smoke inference but is not neural generation.",
    }


def generate_hybrid(
    checkpoint: dict[str, Any],
    *,
    prompt: str,
    teacher_artifacts: str | Path,
    retrieval_threshold: float = 0.45,
    max_tokens: int = 32,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 0,
    prompt_bonus: float = 0.0,
) -> dict[str, Any]:
    nearest = retrieve_nearest_response(prompt=prompt, teacher_artifacts=teacher_artifacts)
    if float(nearest["nearest_score"]) >= retrieval_threshold:
        nearest["status"] = "hybrid_used_nearest"
        nearest["retrieval_threshold"] = retrieval_threshold
        return nearest
    sample = generate_from_token_bias(
        checkpoint,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        prompt_bonus=prompt_bonus,
    )
    sample["status"] = "hybrid_used_token_bias"
    sample["retrieval_threshold"] = retrieval_threshold
    sample["nearest_score"] = nearest["nearest_score"]
    return sample


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run bounded inference/generation for the current sparse-student checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--mode", choices=["token-bias", "nearest", "hybrid"], default="hybrid")
    parser.add_argument("--teacher-artifacts", default=None)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt-bonus", type=float, default=0.0)
    parser.add_argument("--retrieval-threshold", type=float, default=0.45)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args(argv)

    checkpoint = load_student_checkpoint(args.checkpoint)
    if args.mode == "token-bias":
        result = generate_from_token_bias(
            checkpoint,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            prompt_bonus=args.prompt_bonus,
            greedy=args.greedy,
        )
    elif args.mode == "nearest":
        if not args.teacher_artifacts:
            raise SystemExit("--teacher-artifacts is required for nearest mode")
        result = retrieve_nearest_response(prompt=args.prompt, teacher_artifacts=args.teacher_artifacts)
    else:
        if not args.teacher_artifacts:
            raise SystemExit("--teacher-artifacts is required for hybrid mode")
        result = generate_hybrid(
            checkpoint,
            prompt=args.prompt,
            teacher_artifacts=args.teacher_artifacts,
            retrieval_threshold=args.retrieval_threshold,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            prompt_bonus=args.prompt_bonus,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
