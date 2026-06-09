from src.ir import MathNode, var, const, add, matmul, affine, grad, op_class
from src.parser import parse
from src.normalize import normalize


def test_var_is_leaf():
    x = var("x")
    assert x.op == "var"
    assert x.value == "x"
    assert x.arity == 0
    assert x.depth == 0
    assert x.is_leaf()


def test_matmul_arity_depth():
    m = matmul(var("A"), var("x"))
    assert m.op == "matmul"
    assert m.arity == 2
    assert m.depth == 1


def test_nested_depth():
    m = add(matmul(var("A"), var("x")), var("b"))
    assert m.depth == 2


def test_subtree_size():
    m = add(matmul(var("A"), var("x")), var("b"))
    assert m.subtree_size == 5


def test_hash_stable():
    m1 = matmul(var("A"), var("x"))
    m2 = matmul(var("A"), var("x"))
    assert hash(m1) == hash(m2)
    assert m1 == m2


def test_collect_nodes_count():
    m = add(matmul(var("A"), var("x")), var("b"))
    assert len(m.collect_nodes()) == 5


def test_collect_edges_count():
    m = matmul(var("A"), var("x"))
    assert len(m.collect_edges()) == 2


def test_op_class_leaf():
    assert op_class(var("x")) == "leaf"


def test_op_class_matmul():
    assert op_class(matmul(var("A"), var("x"))) == "matmul"


def test_op_class_elementwise():
    assert op_class(add(var("x"), var("y"))) == "elementwise"


def test_op_class_grad():
    assert op_class(grad(var("f"), var("x"))) == "grad"


def test_repr_var():
    assert repr(var("x")) == "x"


def test_repr_matmul():
    assert repr(matmul(var("A"), var("x"))) == "matmul(A, x)"


def test_repr_nested():
    m = add(matmul(var("A"), var("x")), var("b"))
    assert repr(m) == "add(matmul(A, x), b)"


def test_args_are_tuple():
    m = add(var("x"), var("y"))
    assert isinstance(m.args, tuple)


# ── Normalization tests (A2) ──────────────────────────────────────────────────

def test_add_zero_normalizes():
    assert repr(normalize(parse("add(x, 0)"))) == "x"


def test_add_zero_left_normalizes():
    assert repr(normalize(parse("add(0, x)"))) == "x"


def test_mul_one_normalizes():
    assert repr(normalize(parse("mul(x, 1)"))) == "x"


def test_mul_one_left_normalizes():
    assert repr(normalize(parse("mul(1, x)"))) == "x"


def test_matmul_scalar_one_does_not_normalize():
    # matmul identity is NOT scalar 1 — must not be simplified
    assert repr(normalize(parse("matmul(A, 1)"))) == "matmul(A, 1)"


def test_affine_expands():
    result = normalize(parse("affine(A, x, b)"))
    assert repr(result) == "add(matmul(A, x), b)"
