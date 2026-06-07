from __future__ import annotations
import hashlib
import numpy as np
from .ir import MathNode, op_class, COMMUTATIVE_OPS

EMBED_DIM = 64

_OP_IDS: dict[str, int] = {
    "var": 0, "const": 1, "add": 2, "sub": 3, "mul": 4, "div": 5,
    "neg": 6, "matmul": 7, "affine": 8, "sum": 9, "mean": 10,
    "norm": 11, "grad": 12, "constraint": 13, "leq": 14, "geq": 15,
    "eq": 16, "transpose": 17, "inv": 18,
}
_N_OPS = 24  # reserve extra slots

_DTYPE_IDS: dict[str | None, int] = {
    "float32": 0, "float64": 1, "int32": 2, "int64": 3,
    "float16": 4, "bfloat16": 5, None: 6,
}
_N_DTYPES = 8

_OP_CLASS_IDS: dict[str, int] = {
    "leaf": 0, "elementwise": 1, "matmul": 2, "reduction": 3,
    "grad": 4, "constraint": 5, "affine": 6, "generic": 7,
}
_N_CLASSES = 8


def _stable_fingerprint(node: MathNode) -> np.ndarray:
    """8-bit SHA-256 fingerprint — deterministic across processes."""
    digest = hashlib.sha256(repr(node).encode()).digest()
    return np.frombuffer(digest[:8], dtype=np.uint8).astype(np.float32)


class MathEmbedder:
    """Deterministic structural embedder — no learned weights."""

    def encode(self, node: MathNode) -> np.ndarray:
        vec = np.zeros(EMBED_DIM, dtype=np.float32)
        idx = 0

        # Op one-hot (0..N_OPS-1)
        vec[_OP_IDS.get(node.op, _N_OPS - 1)] = 1.0
        idx = _N_OPS

        # Scalar structural features
        vec[idx + 0] = min(node.arity, 8) / 8.0
        vec[idx + 1] = min(node.depth, 16) / 16.0
        vec[idx + 2] = min(node.subtree_size, 64) / 64.0
        vec[idx + 3] = 1.0 if node.op in COMMUTATIVE_OPS else 0.0
        idx += 4

        # Shape (up to 4 dims)
        if node.shape:
            for i, s in enumerate(node.shape[:4]):
                vec[idx + i] = min(s, 2048) / 2048.0
        idx += 4

        # Dtype
        vec[idx] = _DTYPE_IDS.get(node.dtype, _N_DTYPES - 1) / _N_DTYPES
        idx += 1

        # Op-class one-hot
        oc = op_class(node)
        vec[idx + _OP_CLASS_IDS.get(oc, _N_CLASSES - 1)] = 1.0
        idx += _N_CLASSES

        # Stable SHA-256 structural fingerprint (8 bytes)
        fp = _stable_fingerprint(node)
        vec[idx : idx + 8] = fp / 255.0
        idx += 8

        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32) if norm > 0 else vec

    def encode_batch(self, nodes: list[MathNode]) -> np.ndarray:
        return np.stack([self.encode(n) for n in nodes])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pairwise_cosine(Z: np.ndarray) -> np.ndarray:
    """Z: (n, d) -> (n, n) cosine similarity matrix."""
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-8, norms)
    Zn = Z / norms
    return Zn @ Zn.T
