from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import urllib.request
from pathlib import Path

MANIFEST_SCHEMA_VERSION = "qwen_frozen_logit_distillation_targets.v1"
ROW_SCHEMA_VERSION = "qwen_frozen_logit_distillation_target_row.v1"


def checksum_json(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def softmax(logits, temperature: float):
    m = max(logits)
    xs = [math.exp((x - m) / temperature) for x in logits]
    s = sum(xs)
    return [x / s for x in xs]


def parse_seeds(raw: str):
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--seeds must contain at least one integer")
    return values


def generate(model: str, prompt: str, seed: int, timeout: float):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 1,
            "seed": seed,
            "num_ctx": 128,
        },
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seeds", type=parse_seeds, required=True)
    ap.add_argument("--vocab-size", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--label-logit", type=float, default=4.0)
    ap.add_argument("--other-logit", type=float, default=-4.0)
    ap.add_argument("--timeout-seconds", type=float, default=45.0)
    args = ap.parse_args()

    if args.vocab_size < 2:
        raise SystemExit("--vocab-size must be >= 2")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be > 0")
    if args.label_logit <= args.other_logit:
        raise SystemExit("--label-logit must be > --other-logit")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, seed in enumerate(args.seeds):
        prompt = f"Return exactly one integer in [0, {args.vocab_size - 1}]. Only the integer. Seed: {seed}"
        data = generate(args.model, prompt, seed, args.timeout_seconds)
        response = str(data.get("response", "")).strip()
        match = re.search(r"-?\d+", response)
        parsed = False
        if match:
            label = int(match.group(0))
            parsed = 0 <= label < args.vocab_size
        if not parsed:
            label = int(hashlib.sha256(response.encode()).hexdigest()[:12], 16) % args.vocab_size

        logits = [args.other_logit] * args.vocab_size
        logits[label] = args.label_logit
        probabilities = softmax(logits, args.temperature)
        rows.append({
            "schema_version": ROW_SCHEMA_VERSION,
            "row_id": f"ollama_logit_target_{i:06d}",
            "seed": seed,
            "target_type": "logits",
            "vocab_size": args.vocab_size,
            "temperature": args.temperature,
            "logits_checksum": checksum_json(logits),
            "probabilities_checksum": checksum_json(probabilities),
            "logits": logits,
            "probabilities": probabilities,
            "metadata": {
                "target_mode": "ollama_sampled_hard_label_pseudo_logits",
                "teacher_model": args.model,
                "teacher_checkpoint_loaded_at_runtime": False,
                "teacher_inference_runtime_required": False,
                "ollama_runtime_used_offline": True,
                "selected_label": label,
                "label_parse_ok": parsed,
                "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                "total_duration_ns": data.get("total_duration"),
                "load_duration_ns": data.get("load_duration"),
                "eval_count": data.get("eval_count"),
            },
        })

    rows_path = out / "frozen_logit_targets.jsonl"
    rows_path.write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "frozen_logit_targets_ready",
        "target_type": "logits",
        "producer": "offline_ollama_sampled_teacher",
        "teacher_model": args.model,
        "teacher_checkpoint_loaded_at_runtime": False,
        "teacher_inference_runtime_required": False,
        "raw_weight_payload_in_graph": False,
        "student_training_started": False,
        "kl_training_started": False,
        "vocab_size": args.vocab_size,
        "row_count": len(rows),
        "target_rows_path": "frozen_logit_targets.jsonl",
        "target_rows_sha256": sha256_file(rows_path),
        "temperature": args.temperature,
        "promotion_eligible": False,
        "target_mode": "ollama_sampled_hard_label_pseudo_logits",
        "label_logit": args.label_logit,
        "other_logit": args.other_logit,
    }
    (out / "frozen_logit_targets_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "teacher_model": args.model, "labels": [r["metadata"]["selected_label"] for r in rows], "target_rows_sha256": manifest["target_rows_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
