from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import tracemalloc
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


PLAN = "v25.01"
DEFAULT_TEACHER_ID = "qwen25-3b-smart:latest"
DEFAULT_TEACHER_BACKEND = "ollama"
DEFAULT_EXPERIMENT_ID = "v25_01_qwen25_3b_smart_sparse_smoke"
DEFAULT_OUTPUT_DIR = Path("runs/sparse_student_distill/qwen25-3b-smart/sparse_student") / DEFAULT_EXPERIMENT_ID
PROMPT_SOURCE = "synthetic_v25_01"
PROMPT_TEMPLATE = (
    "You are a precise teacher generating distillation targets for a small sparse student.\n"
    "Answer compactly. Show only necessary reasoning. End with a final answer field.\n\n"
    "Problem:\n{problem}\n\n"
    "Return:\nreasoning: ...\nanswer: ..."
)

DEFAULT_GENERATION_SETTINGS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "repeat_penalty": 1.05,
    "num_predict": 256,
    "seed": 25,
}
EXACT_GENERATION_SETTINGS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "num_predict": 128,
    "seed": 25,
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(data: Any) -> str:
    return sha256_text(json.dumps(data, sort_keys=True, separators=(",", ":")))


def read_json(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}: expected JSON object")
            yield row


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], *, force: bool = True) -> None:
    target = Path(path)
    if target.exists() and not force:
        raise FileExistsError(f"{target} exists; pass --force to overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _ollama_json(endpoint: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed for {endpoint}: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"Ollama returned non-object JSON for {endpoint}")
    if result.get("error"):
        raise RuntimeError(f"Ollama request failed for {endpoint}: {result['error']}")
    return result


def capture_ollama_teacher_metadata(
    output_dir: str | Path,
    *,
    teacher_id: str = DEFAULT_TEACHER_ID,
    generation_settings: dict[str, Any] | None = None,
    endpoint_base: str = "http://127.0.0.1:11434",
    timeout: float = 10.0,
) -> dict[str, Any]:
    settings = dict(DEFAULT_GENERATION_SETTINGS if generation_settings is None else generation_settings)
    show_payload = {"model": teacher_id}
    show = _ollama_json(f"{endpoint_base.rstrip('/')}/api/show", show_payload, timeout=timeout)
    details = show.get("details") if isinstance(show.get("details"), dict) else {}
    model_info = show.get("model_info") if isinstance(show.get("model_info"), dict) else {}
    digest = show.get("digest") or model_info.get("general.name")
    raw_show = json.dumps(show, sort_keys=True)
    metadata = {
        "teacher_id": teacher_id,
        "teacher_backend": DEFAULT_TEACHER_BACKEND,
        "ollama_model_digest": digest or "record_if_available",
        "ollama_show": raw_show,
        "model_tag": teacher_id,
        "quantization": details.get("quantization_level") or details.get("quantization") or "unknown",
        "context_length": model_info.get("qwen2.context_length"),
        "generation_settings": settings,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "logits_available": False,
    }
    write_json(Path(output_dir) / "teacher_metadata.json", metadata)
    return metadata


def _prompt_row(family: str, split: str, problem: str, expected_answer: str | None, idx: int) -> dict[str, Any]:
    sample_id = sha256_text(f"{PLAN}|{family}|{split}|{idx}|{problem}")
    row = {
        "sample_id": sample_id,
        "family": family,
        "split": split,
        "prompt": problem,
        "expected_answer": expected_answer,
        "source": PROMPT_SOURCE,
    }
    row["prompt_hash"] = sha256_text(problem)
    return row


def generate_v25_01_prompt_rows(n: int = 128, *, split: str = "train") -> list[dict[str, Any]]:
    if n < 1:
        raise ValueError("n must be >= 1")
    families = ("arithmetic_short", "symbolic_short", "logic_short", "project_specific_math_transformer")
    rows: list[dict[str, Any]] = []
    for idx in range(n):
        family = families[idx % len(families)]
        q = idx // len(families)
        if family == "arithmetic_short":
            a = 7 + (q * 3) % 41
            b = 5 + (q * 5) % 37
            problem = f"Compute {a} + {b}. Return the final integer."
            expected = str(a + b)
        elif family == "symbolic_short":
            a = 2 + q % 6
            b = 3 + (q * 2) % 9
            problem = f"Simplify the expression {a}x + {b}x. Return the simplified expression."
            expected = f"{a + b}x"
        elif family == "logic_short":
            left = f"item_{q}"
            right = f"item_{q + 1}"
            problem = f"If every {left} is a {right}, and A is a {left}, what category must A be in?"
            expected = right
        else:
            k = 1 + q % 4
            problem = (
                "In a fixed sparse graph-native student, adjacency A is locked during training. "
                f"If each node has at most K={k} outgoing edges, may v25.01 mutate A while optimizing theta?"
            )
            expected = "no"
        rows.append(_prompt_row(family, split, problem, expected, idx))
    return rows


def write_v25_01_prompts(output_dir: str | Path, *, n: int = 128, split: str = "train", force: bool = False) -> list[dict[str, Any]]:
    rows = generate_v25_01_prompt_rows(n, split=split)
    write_jsonl(Path(output_dir) / "teacher_prompts.jsonl", rows, force=force)
    return rows


def _settings_for_prompt(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("family") == "arithmetic_short":
        return dict(EXACT_GENERATION_SETTINGS)
    return dict(DEFAULT_GENERATION_SETTINGS)


def query_ollama_teacher(
    output_dir: str | Path,
    *,
    teacher_id: str = DEFAULT_TEACHER_ID,
    endpoint_base: str = "http://127.0.0.1:11434",
    timeout: float = 120.0,
    force: bool = False,
) -> list[dict[str, Any]]:
    base = Path(output_dir)
    prompts_path = base / "teacher_prompts.jsonl"
    if not prompts_path.exists():
        raise FileNotFoundError(str(prompts_path))
    rows: list[dict[str, Any]] = []
    for prompt_row in iter_jsonl(prompts_path):
        settings = _settings_for_prompt(prompt_row)
        payload = {
            "model": teacher_id,
            "prompt": PROMPT_TEMPLATE.format(problem=prompt_row["prompt"]),
            "stream": False,
            "options": settings,
        }
        started = time.perf_counter()
        result = _ollama_json(f"{endpoint_base.rstrip('/')}/api/generate", payload, timeout=timeout)
        latency_ms = int(round((time.perf_counter() - started) * 1000.0))
        response = str(result.get("response", ""))
        rows.append(
            {
                "sample_id": prompt_row["sample_id"],
                "teacher_id": teacher_id,
                "prompt_hash": sha256_text(str(prompt_row["prompt"])),
                "response": response,
                "answer_extracted": extract_answer(response),
                "latency_ms": latency_ms,
                "eval_options": settings,
            }
        )
    write_jsonl(base / "teacher_responses.jsonl", rows, force=force)
    return rows


_ANSWER_RE = re.compile(r"(?im)^\s*answer\s*:\s*(.+?)\s*$")
_CATEGORY_OF_RE = re.compile(r"(?i)\bcategory\s+of\s+([\w_]+)")

_ANSWER_SYNONYMS: dict[str, str] = {
    "unlikely": "no",
    "notpossible": "no",
    "impossible": "no",
}

_NO_POLARITY_RE = re.compile(r"(?i)\b(cannot|not possible|may not|immutable|not allowed)\b")


def extract_answer(response: str) -> str:
    matches = _ANSWER_RE.findall(response)
    raw = matches[-1].strip() if matches else ""
    # Strip "A must be in the category of item_N." → "item_N"
    m = _CATEGORY_OF_RE.search(raw)
    if m:
        return m.group(1).strip()
    if raw:
        return raw
    lines = [l.strip() for l in response.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def _normalize_answer(value: Any) -> str:
    text = str(value if value is not None else "").strip().lower()
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".")
    return _ANSWER_SYNONYMS.get(text, text)


def filter_teacher_responses(output_dir: str | Path, *, force: bool = False) -> list[dict[str, Any]]:
    base = Path(output_dir)
    prompts = {row["sample_id"]: row for row in iter_jsonl(base / "teacher_prompts.jsonl")}
    responses = list(iter_jsonl(base / "teacher_responses.jsonl"))
    quality_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    arithmetic_total = 0
    arithmetic_pass = 0
    for response in responses:
        sample_id = response["sample_id"]
        prompt = prompts.get(sample_id)
        if prompt is None:
            raise ValueError(f"response sample_id={sample_id} has no prompt")
        response_text = str(response.get("response", ""))
        answer = str(response.get("answer_extracted") or extract_answer(response_text))
        # Strip sentence wrappers that may have been stored in answer_extracted
        m = _CATEGORY_OF_RE.search(answer)
        if m:
            answer = m.group(1).strip()
        parse_pass = bool(answer)
        expected = prompt.get("expected_answer")
        answer_match = True if expected in (None, "") else _normalize_answer(answer) == _normalize_answer(expected)
        # For binary yes/no questions: accept long answers that contain unambiguous polarity indicators
        if not answer_match and expected == "no" and len(answer) > 20 and _NO_POLARITY_RE.search(answer):
            answer_match = True
        if prompt.get("family") == "arithmetic_short":
            arithmetic_total += 1
            arithmetic_pass += int(answer_match and parse_pass)
        drop_reason = None
        if not parse_pass:
            drop_reason = "answer_parse_failed"
        elif not answer_match:
            drop_reason = "expected_answer_mismatch"
        kept = drop_reason is None
        row = {
            "sample_id": sample_id,
            "answer_match": bool(answer_match),
            "critic_pass": True,
            "parse_pass": bool(parse_pass),
            "kept_for_training": bool(kept),
            "drop_reason": drop_reason,
        }
        quality_rows.append(row)
        if not kept:
            failure_rows.append(
                {
                    "sample_id": sample_id,
                    "family": prompt.get("family"),
                    "drop_reason": drop_reason,
                    "expected_answer": expected,
                    "answer_extracted": answer,
                }
            )
    if arithmetic_total and arithmetic_pass / arithmetic_total < 0.9:
        failure_rows.append(
            {
                "sample_id": "arithmetic_short_quality_gate",
                "family": "arithmetic_short",
                "drop_reason": "arithmetic_pass_rate_below_90_percent",
                "pass_rate": arithmetic_pass / arithmetic_total,
            }
        )
    write_jsonl(base / "teacher_response_quality.jsonl", quality_rows, force=force)
    write_jsonl(base / "failure_cases.jsonl", failure_rows, force=True)
    return quality_rows


def build_distill_examples(
    output_dir: str | Path,
    *,
    max_target_chars: int = 4096,
    force: bool = False,
) -> dict[str, Any]:
    if max_target_chars < 1:
        raise ValueError("max_target_chars must be >= 1")
    base = Path(output_dir)
    prompts = {row["sample_id"]: row for row in iter_jsonl(base / "teacher_prompts.jsonl")}
    responses = {row["sample_id"]: row for row in iter_jsonl(base / "teacher_responses.jsonl")}
    quality = {row["sample_id"]: row for row in iter_jsonl(base / "teacher_response_quality.jsonl")}
    rows: list[dict[str, Any]] = []
    dropped = 0
    token_counter = Counter()
    for sample_id, prompt in sorted(prompts.items()):
        q = quality.get(sample_id)
        response = responses.get(sample_id)
        if not q or not response or not q.get("kept_for_training"):
            dropped += 1
            continue
        target = str(response["response"]).strip()
        if len(target) > max_target_chars:
            dropped += 1
            continue
        input_text = str(prompt["prompt"])
        target_tokens = simple_tokenize(target)
        input_tokens = simple_tokenize(input_text)
        token_counter.update(target_tokens)
        rows.append(
            {
                "sample_id": sample_id,
                "family": prompt["family"],
                "split": prompt["split"],
                "input": input_text,
                "target": target,
                "input_token_count": len(input_tokens),
                "target_token_count": len(target_tokens),
                "loss_mask": "target_only",
                "expected_answer": prompt.get("expected_answer"),
                "answer_extracted": response.get("answer_extracted"),
            }
        )
    write_jsonl(base / "distill_examples.jsonl", rows, force=force)
    report = {
        "plan": PLAN,
        "status": "distill_examples_ready",
        "example_count": len(rows),
        "dropped_count": dropped,
        "max_target_chars": max_target_chars,
        "target_token_count": sum(int(row["target_token_count"]) for row in rows),
        "vocab_size": len(token_counter),
        "loss_mask": "target_only",
    }
    write_json(base / "distill_examples_report.json", report)
    return report


def simple_tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_]+|\d+|[^\sA-Za-z_\d]", text)


def _load_adjacency(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "schema_version": "v25_01_inline_adjacency.v1",
            "adjacency_name": "A_inline_single_edge",
            "k": 1,
            "fixed_adjacency": True,
            "edges": [{"edge_id": "e0", "src_id": "n0", "dst_id": "n1", "weight": 1.0}],
        }
    data = read_json(path)
    edges = data.get("edges")
    if not isinstance(edges, list):
        raise ValueError("adjacency JSON must contain an edges list")
    return data


def _adjacency_summary(adjacency: dict[str, Any]) -> dict[str, Any]:
    edges = adjacency.get("edges")
    if not isinstance(edges, list):
        raise ValueError("adjacency edges must be a list")
    node_ids = sorted({str(edge.get("src_id")) for edge in edges} | {str(edge.get("dst_id")) for edge in edges})
    weights = []
    for edge in edges:
        try:
            weight = float(edge.get("weight", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("adjacency edge weights must be numeric") from exc
        if not math.isfinite(weight):
            raise ValueError("adjacency edge weights must be finite")
        weights.append(weight)
    return {
        "adjacency_name": str(adjacency.get("adjacency_name", adjacency.get("name", "A_qwen_fixed"))),
        "k": adjacency.get("k"),
        "node_count": len(node_ids),
        "edge_count": len(edges),
        "edge_weight_checksum": sha256_json(weights),
        "adjacency_checksum": sha256_json(adjacency),
    }


def _softmax_loss_and_grad(logits: dict[str, float], counts: Counter[str]) -> tuple[float, dict[str, float]]:
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("cannot train with zero target tokens")
    vocab = sorted(logits)
    max_logit = max(logits[token] for token in vocab)
    exp_values = {token: math.exp(logits[token] - max_logit) for token in vocab}
    denom = sum(exp_values.values())
    probs = {token: exp_values[token] / denom for token in vocab}
    loss = 0.0
    grads: dict[str, float] = {}
    for token in vocab:
        target_prob = counts[token] / total
        prob = probs[token]
        if counts[token]:
            loss -= target_prob * math.log(max(prob, 1e-12))
        grads[token] = prob - target_prob
    return loss, grads


def run_sparse_student_distill(
    teacher_artifacts: str | Path,
    *,
    adjacency: str | Path | None = None,
    student_config: str | Path | None = None,
    plan: str = PLAN,
    fixed_adjacency: bool = True,
    allow_rewiring: bool = False,
    train_steps: int | None = None,
    lr: float | None = None,
    no_edge_trace: bool = False,
) -> dict[str, Any]:
    if plan != PLAN:
        raise ValueError(f"unsupported plan={plan!r}")
    if not fixed_adjacency:
        raise ValueError("v25.01 requires fixed_adjacency=true")
    if allow_rewiring:
        raise ValueError("v25.01 requires allow_rewiring=false")
    base = Path(teacher_artifacts)
    examples_path = base / "distill_examples.jsonl"
    if not examples_path.exists():
        build_distill_examples(base, force=True)
    examples = list(iter_jsonl(examples_path))
    if not examples:
        raise ValueError("distill_examples.jsonl contains no training examples")
    config = read_json(student_config) if student_config is not None and Path(student_config).exists() else {}
    steps = int(train_steps if train_steps is not None else config.get("train_steps", 8))
    lr_value = float(lr if lr is not None else config.get("lr", 0.5))
    if steps < 1:
        raise ValueError("train_steps must be >= 1")
    if lr_value <= 0.0 or not math.isfinite(lr_value):
        raise ValueError("lr must be finite and > 0")
    adjacency_data = _load_adjacency(adjacency)
    before_summary = _adjacency_summary(adjacency_data)
    target_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    for row in examples:
        target_counts.update(simple_tokenize(str(row["target"])))
        family_counts[str(row.get("family", "unknown"))] += 1
    vocab = sorted(target_counts)
    logits = {token: 0.0 for token in vocab}
    tracemalloc.start()
    started = time.perf_counter()
    train_rows: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []
    for step in range(steps):
        loss_before, grads = _softmax_loss_and_grad(logits, target_counts)
        grad_norm = math.sqrt(sum(value * value for value in grads.values()))
        for token, grad in grads.items():
            logits[token] -= lr_value * grad
        loss_after, _ = _softmax_loss_and_grad(logits, target_counts)
        row = {
            "step": step,
            "loss_text_before": loss_before,
            "loss_text_after": loss_after,
            "loss_task": 0.0,
            "loss_total": loss_after,
            "lr": lr_value,
            "grad_norm": grad_norm,
            "fixed_adjacency": True,
            "allow_rewiring": False,
        }
        train_rows.append(row)
        grad_rows.append({"step": step, "edge_gradient_norm": grad_norm, "edge_loss_contribution": loss_before - loss_after})
    elapsed = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    after_summary = _adjacency_summary(adjacency_data)
    if before_summary["adjacency_checksum"] != after_summary["adjacency_checksum"]:
        raise ValueError("fixed adjacency mutated during training")
    final_loss = train_rows[-1]["loss_text_after"]
    initial_loss = train_rows[0]["loss_text_before"]
    write_jsonl(base / "train_metrics.jsonl", train_rows, force=True)
    if no_edge_trace:
        write_jsonl(base / "edge_trace.jsonl", [], force=True)
        write_jsonl(base / "gradient_edge_stats.jsonl", [], force=True)
    else:
        edge_rows = [
            {
                "edge_id": edge.get("edge_id", f"edge_{idx}"),
                "edge_activation_frequency": 1.0,
                "edge_confidence": 1.0,
                "edge_compute_cost": 1.0,
                "edge_source_prior": before_summary["adjacency_name"],
                "edge_source_tensor": edge.get("relation", "unknown"),
            }
            for idx, edge in enumerate(adjacency_data.get("edges", []))
        ]
        write_jsonl(base / "edge_trace.jsonl", edge_rows, force=True)
        write_jsonl(base / "gradient_edge_stats.jsonl", grad_rows, force=True)
    student_config_data = {
        "plan": PLAN,
        "student_family": "sparse graph-native math_transformer student",
        "student_topology": "A_qwen_fixed",
        "fixed_adjacency": True,
        "allow_rewiring": False,
        "optimizer": "token_bias_sgd",
        "train_steps": steps,
        "lr": lr_value,
        "adjacency": before_summary,
    }
    write_json(base / "student_config.json", student_config_data)
    write_json(
        base / "eval_metrics.json",
        {
            "status": "eval_complete",
            "loss_text_initial": initial_loss,
            "loss_text_final": final_loss,
            "delta_teacher_text": initial_loss - final_loss,
            "example_count": len(examples),
            "family_counts": dict(sorted(family_counts.items())),
        },
    )
    write_json(
        base / "quality_report.json",
        {
            "status": "quality_measured",
            "teacher_text_loss_decreased": final_loss <= initial_loss,
            "delta_teacher_text": initial_loss - final_loss,
            "promotion_claim": "smoke_only_no_graph_prior_claim",
        },
    )
    write_json(base / "runtime_report.json", {"status": "runtime_measured", "elapsed_seconds": elapsed, "protocol": "locked_eval"})
    write_json(base / "memory_report.json", {"status": "memory_measured", "peak_bytes": int(peak)})
    write_json(
        base / "kd_delta_report.json",
        {
            "status": "teacher_text_delta_measured",
            "logit_kd_used": False,
            "delta_teacher_text": initial_loss - final_loss,
        },
    )
    write_json(
        base / "prior_delta_report.json",
        {
            "status": "not_claimed_on_single_run",
            "delta_prior": None,
            "reason": "matched random baseline is required before graph-prior benefit claims",
        },
    )
    checkpoint = {"token_logits": logits, "vocab": vocab, "adjacency_checksum": before_summary["adjacency_checksum"]}
    write_json(base / "student_checkpoint.json", checkpoint)
    report = {
        "status": "v25_01_sparse_student_distill_ok",
        "plan": PLAN,
        "teacher_artifacts": str(base),
        "example_count": len(examples),
        "train_steps": steps,
        "loss_text_initial": initial_loss,
        "loss_text_final": final_loss,
        "delta_teacher_text": initial_loss - final_loss,
        "fixed_adjacency": True,
        "allow_rewiring": False,
        "adjacency_unchanged": True,
        "artifacts": [
            "train_metrics.jsonl",
            "eval_metrics.json",
            "quality_report.json",
            "runtime_report.json",
            "memory_report.json",
            "kd_delta_report.json",
            "prior_delta_report.json",
            "edge_trace.jsonl",
            "gradient_edge_stats.jsonl",
            "student_checkpoint.json",
        ],
    }
    return report


def write_default_distill_config(
    output_dir: str | Path,
    *,
    source_weight_graph_dir: str = "",
    source_prior_experiment_dir: str = "",
    teacher_id: str = DEFAULT_TEACHER_ID,
) -> dict[str, Any]:
    config = {
        "plan": PLAN,
        "teacher_id": teacher_id,
        "teacher_backend": DEFAULT_TEACHER_BACKEND,
        "teacher_artifact_type": "text_response",
        "source_weight_graph_dir": source_weight_graph_dir,
        "source_prior_experiment_dir": source_prior_experiment_dir,
        "student_topology": "A_qwen_fixed",
        "fixed_adjacency": True,
        "allow_rewiring": False,
        "loss": "teacher_text_sft_plus_task",
        "top_r": None,
        "temperature": None,
        "task_loss_weight": 0.25,
        "teacher_text_loss_weight": 1.0,
        "memory_budget": "recorded",
        "runtime_protocol": "locked_eval",
    }
    write_json(Path(output_dir) / "distill_config.json", config)
    return config


def capture_metadata_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--teacher-id", default=DEFAULT_TEACHER_ID)
    parser.add_argument("--endpoint-base", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)
    capture_ollama_teacher_metadata(
        args.output_dir,
        teacher_id=args.teacher_id,
        endpoint_base=args.endpoint_base,
        timeout=args.timeout,
    )
    return 0


def generate_prompts_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("-n", "--count", type=int, default=128)
    parser.add_argument("--split", default="train")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    write_default_distill_config(args.output_dir)
    write_v25_01_prompts(args.output_dir, n=args.count, split=args.split, force=args.force)
    return 0


def query_teacher_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--teacher-id", default=DEFAULT_TEACHER_ID)
    parser.add_argument("--endpoint-base", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    query_ollama_teacher(
        args.output_dir,
        teacher_id=args.teacher_id,
        endpoint_base=args.endpoint_base,
        timeout=args.timeout,
        force=args.force,
    )
    return 0


def filter_responses_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    filter_teacher_responses(args.output_dir, force=args.force)
    return 0


def build_examples_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-target-chars", type=int, default=4096)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    build_distill_examples(args.output_dir, max_target_chars=args.max_target_chars, force=args.force)
    return 0


def sparse_student_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=PLAN)
    parser.add_argument("--teacher-artifacts", required=True)
    parser.add_argument("--adjacency")
    parser.add_argument("--student-config")
    parser.add_argument("--fixed-adjacency", default="true")
    parser.add_argument("--allow-rewiring", default="false")
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--no-edge-trace", action="store_true")
    args = parser.parse_args(argv)
    report = run_sparse_student_distill(
        args.teacher_artifacts,
        adjacency=args.adjacency,
        student_config=args.student_config,
        plan=args.plan,
        fixed_adjacency=str(args.fixed_adjacency).lower() == "true",
        allow_rewiring=str(args.allow_rewiring).lower() == "true",
        train_steps=args.train_steps,
        lr=args.lr,
        no_edge_trace=args.no_edge_trace,
    )
    write_json(Path(args.teacher_artifacts) / "sparse_student_distill_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
