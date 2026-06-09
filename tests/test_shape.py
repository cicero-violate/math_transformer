import pytest
from src.ir import var, const, add, matmul, affine, grad
from src.parser import parse
from src.normalize import normalize
from src.shape import infer_shape, infer_tree, ShapeError


def test_var_with_env():
    assert infer_shape(var("x"), {"x": (32,)}) == (32,)


def test_var_missing_env():
    assert infer_shape(var("x"), {}) is None


def test_const_scalar():
    assert infer_shape(const(0)) == ()


def test_const_with_shape():
    from src.ir import MathNode
    node = MathNode(op="const", args=(), value=1.0, shape=(3,))
    assert infer_shape(node) == (3,)


def test_add_same_shapes():
    env = {"x": (32,), "y": (32,)}
    assert infer_shape(parse("add(x, y)"), env) == (32,)


def test_add_shape_mismatch_raises():
    env = {"x": (32,), "y": (64,)}
    with pytest.raises(ShapeError):
        infer_shape(parse("add(x, y)"), env)


def test_add_scalar_broadcast():
    env = {"x": (32,)}
    node = add(var("x"), const(0))
    assert infer_shape(node, env) == (32,)


def test_matmul_matrix_vector():
    env = {"A": (32, 64), "x": (64,)}
    assert infer_shape(parse("matmul(A, x)"), env) == (32,)


def test_matmul_matrix_matrix():
    env = {"A": (32, 64), "B": (64, 16)}
    assert infer_shape(parse("matmul(A, B)"), env) == (32, 16)


def test_matmul_inner_dim_mismatch_raises():
    env = {"A": (32, 64), "x": (32,)}
    with pytest.raises(ShapeError):
        infer_shape(parse("matmul(A, x)"), env)


def test_affine_shape():
    env = {"A": (32, 64), "x": (64,), "b": (32,)}
    result = infer_shape(normalize(parse("affine(A, x, b)")), env)
    assert result == (32,)


def test_affine_bias_mismatch_raises():
    env = {"A": (32, 64), "x": (64,), "b": (16,)}
    with pytest.raises(ShapeError):
        infer_shape(normalize(parse("affine(A, x, b)")), env)


def test_nested_matmul():
    # add(matmul(A, x), b) with compatible shapes
    env = {"A": (32, 64), "x": (64,), "b": (32,)}
    node = normalize(parse("affine(A, x, b)"))
    assert infer_shape(node, env) == (32,)


def test_infer_shape_unknown_op():
    node = parse("grad(f, x)")
    env = {"x": (32,)}
    # grad(f, x) → same shape as x
    result = infer_shape(node, env)
    assert result == (32,)


def test_infer_tree_all_nodes():
    root = normalize(parse("affine(A, x, b)"))
    env = {"A": (32, 64), "x": (64,), "b": (32,)}
    shapes = infer_tree(root, env)
    assert len(shapes) == len(root.collect_nodes())


def test_infer_tree_raises_on_bad_shape():
    root = normalize(parse("affine(A, x, b)"))
    env = {"A": (32, 64), "x": (32,), "b": (32,)}  # inner-dim mismatch
    with pytest.raises(ShapeError):
        infer_tree(root, env)
