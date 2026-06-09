"""Sprint 1 — env-aware model forward tests (Gate B)."""
import torch
import pytest
from src.parser import parse
from src.normalize import normalize
from src.model import MathRoutedTransformer

EXPRS = [
    "add(matmul(A, x), b)",
    "add(matmul(W, h), c)",
    "matmul(Q, K)",
    "grad(f, x)",
]
ENV = {"A": (32, 64), "x": (64,), "b": (32,), "W": (32, 64), "h": (64,),
       "c": (32,), "Q": (8, 16), "K": (16, 8), "f": (32,)}


def _nodes(n: int = 8):
    roots = [normalize(parse(e)) for e in EXPRS]
    nodes = []
    while len(nodes) < n:
        nodes.extend(roots[:n - len(nodes)])
    return nodes[:n]


def test_model_forward_with_env_shape():
    T, D = 8, 64
    nodes = _nodes(T)
    model = MathRoutedTransformer(d_model=D, n_heads=4, n_layers=1, d_ff=128,
                                  topk=2, local_window=1, attention_mode="dense_masked")
    x = model.embed_nodes(nodes)
    with torch.no_grad():
        out, masks, routes = model(x, nodes, env=ENV)
    assert out.shape == (1, T, D)
    assert masks[0] is not None


def test_model_return_diagnostics_with_env():
    T, D = 8, 64
    nodes = _nodes(T)
    model = MathRoutedTransformer(d_model=D, n_heads=4, n_layers=1, d_ff=128,
                                  topk=2, local_window=1, attention_mode="dense_masked")
    x = model.embed_nodes(nodes)
    with torch.no_grad():
        result = model(x, nodes, env=ENV, return_diagnostics=True)
    assert len(result) == 4
    out, masks, routes, diags = result
    assert diags[0] is not None


def test_env_activates_shape_compat_and_composition():
    T, D = 8, 64
    nodes = _nodes(T)
    model = MathRoutedTransformer(d_model=D, n_heads=4, n_layers=1, d_ff=128,
                                  topk=2, local_window=1, attention_mode="dense_masked")
    x = model.embed_nodes(nodes)
    with torch.no_grad():
        _, _, _, diags = model(x, nodes, env=ENV, return_diagnostics=True)

    diag = diags[0]
    assert diag.by_relation["shape_compat"] > 0, \
        f"Expected shape_compat > 0 with env, got {diag.by_relation['shape_compat']}"
    assert diag.by_relation["composition"] > 0, \
        f"Expected composition > 0 with env, got {diag.by_relation['composition']}"


def test_no_env_gives_zero_shape_compat():
    T, D = 8, 64
    nodes = _nodes(T)
    model = MathRoutedTransformer(d_model=D, n_heads=4, n_layers=1, d_ff=128,
                                  topk=2, local_window=1, attention_mode="dense_masked")
    x = model.embed_nodes(nodes)
    with torch.no_grad():
        _, _, _, diags = model(x, nodes, env=None, return_diagnostics=True)
    diag = diags[0]
    assert diag.by_relation["shape_compat"] == 0
    assert diag.by_relation["composition"] == 0


def test_neighbor_sparse_with_env_shape():
    T, D = 8, 64
    nodes = _nodes(T)
    model = MathRoutedTransformer(d_model=D, n_heads=4, n_layers=1, d_ff=128,
                                  topk=2, local_window=1, attention_mode="neighbor_sparse")
    x = model.embed_nodes(nodes)
    with torch.no_grad():
        out, masks, routes = model(x, nodes, env=ENV)
    assert out.shape == (1, T, D)


def test_neighbor_sparse_diags_with_env():
    T, D = 8, 64
    nodes = _nodes(T)
    model = MathRoutedTransformer(d_model=D, n_heads=4, n_layers=1, d_ff=128,
                                  topk=2, local_window=1, attention_mode="neighbor_sparse")
    x = model.embed_nodes(nodes)
    with torch.no_grad():
        _, _, _, diags = model(x, nodes, env=ENV, return_diagnostics=True)
    diag = diags[0]
    assert diag.by_relation["shape_compat"] > 0
