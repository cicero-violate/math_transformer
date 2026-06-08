import subprocess
import sys
from pathlib import Path
import yaml

PROJ = Path(__file__).parent.parent  # math_transformer/


def test_resolve_data_path_logic():
    from src.train import resolve_data_path
    config_path = str(PROJ / "configs" / "tiny.yaml")
    resolved = resolve_data_path(config_path, "data/examples.jsonl")
    assert resolved == PROJ / "data" / "examples.jsonl"
    assert resolved.exists(), f"Data file not found: {resolved}"


def test_resolve_data_path_from_subdirectory():
    """Path resolution should be independent of cwd."""
    from src.train import resolve_data_path
    config_path = str(PROJ / "configs" / "debug.yaml")
    resolved = resolve_data_path(config_path, "data/examples.jsonl")
    assert resolved.exists()


def test_stable_fingerprint_across_processes():
    """SHA-256 fingerprints must be identical across separate Python processes."""
    script = (
        "import sys; sys.path.insert(0, '.'); "
        "from src.embedder import MathEmbedder; "
        "from src.ir import matmul, var; "
        "e = MathEmbedder(); "
        "v = e.encode(matmul(var('A'), var('x'))); "
        "print(','.join(f'{x:.8f}' for x in v[:4].tolist()))"
    )
    results = set()
    for _ in range(2):
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True,
            cwd=str(PROJ),
        )
        assert r.returncode == 0, f"Subprocess failed:\n{r.stderr}"
        results.add(r.stdout.strip())

    assert len(results) == 1, f"Fingerprint differs across processes: {results}"


def test_train_can_save_checkpoint(tmp_path):
    from src.train import train

    cfg = {
        "model": {
            "d_model": 16,
            "n_heads": 2,
            "n_layers": 1,
            "d_ff": 32,
            "dropout": 0.0,
            "topk": 1,
            "local_window": 1,
        },
        "training": {
            "batch_size": 1,
            "lr": 1.0e-3,
            "max_steps": 1,
            "warmup_steps": 0,
        },
        "data": {
            "path": str(PROJ / "data" / "examples.jsonl"),
            "max_seq_len": 32,
        },
        "eval": {
            "interval": 1,
            "metrics": ["route_accuracy"],
        },
    }
    cfg_path = tmp_path / "tiny_train.yaml"
    ckpt_path = tmp_path / "model.pt"
    cfg_path.write_text(yaml.safe_dump(cfg))

    train(str(cfg_path), save_checkpoint=str(ckpt_path))

    assert ckpt_path.exists()
