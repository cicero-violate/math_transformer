from src.ir import var, grad
from src.parser import parse
from src.normalize import normalize
from src.router import OperatorRouter


def test_route_matmul():
    node = parse("matmul(A, B)")
    result = OperatorRouter().route(node)
    assert result.expert == "matmul_expert"


def test_route_affine_form():
    node = normalize(parse("affine(A, x, b)"))
    result = OperatorRouter().route(node)
    assert result.expert == "affine_expert"


def test_route_add_matmul_is_affine():
    node = parse("add(matmul(W, x), b)")
    result = OperatorRouter().route(node)
    assert result.expert == "affine_expert"


def test_route_sum():
    node = parse("sum(i, x_i)")
    result = OperatorRouter().route(node)
    assert result.expert == "reduction_expert"


def test_route_mean():
    node = parse("mean(i, x_i)")
    result = OperatorRouter().route(node)
    assert result.expert == "reduction_expert"


def test_route_grad():
    node = parse("grad(f, x)")
    result = OperatorRouter().route(node)
    assert result.expert == "grad_expert"


def test_route_constraint():
    node = parse("constraint(leq(matmul(A, x), b))")
    result = OperatorRouter().route(node)
    assert result.expert == "constraint_expert"


def test_route_var_is_generic():
    result = OperatorRouter().route(var("x"))
    assert result.expert == "generic_expert"


def test_route_batch_length():
    nodes = [
        parse("matmul(A, B)"),
        normalize(parse("affine(A, x, b)")),
        parse("grad(f, x)"),
    ]
    results = OperatorRouter().route_batch(nodes)
    assert len(results) == 3
    experts = [r.expert for r in results]
    assert experts == ["matmul_expert", "affine_expert", "grad_expert"]


def test_diagnostics_counts():
    nodes = [parse("matmul(A, B)"), parse("matmul(Q, K)"), parse("grad(f, x)")]
    diag = OperatorRouter().route_diagnostics(nodes)
    assert diag["total"] == 3
    assert diag["counts"]["matmul_expert"] == 2
    assert diag["counts"]["grad_expert"] == 1
