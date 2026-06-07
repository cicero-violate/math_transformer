from src.parser import parse
from src.normalize import normalize
from src.verifier import Verifier, ExecutionPlan


def _plan(node, expert, input_shapes=None, input_dtypes=None):
    return ExecutionPlan(
        node=node,
        expert=expert,
        input_shapes=input_shapes or {},
        input_dtypes=input_dtypes or {},
    )


def test_valid_matmul_shapes():
    node = parse("matmul(A, x)")
    plan = _plan(node, "matmul_expert", {"A": (32, 64), "x": (64,)})
    assert Verifier().check(node, plan).passed


def test_invalid_matmul_shape_mismatch():
    node = parse("matmul(A, x)")
    plan = _plan(node, "matmul_expert", {"A": (32, 64), "x": (32,)})
    result = Verifier().check(node, plan)
    assert not result.passed
    assert result.level == 0


def test_valid_add_same_shapes():
    node = parse("add(x, y)")
    plan = _plan(node, "generic_expert", {"x": (32,), "y": (32,)})
    assert Verifier().check(node, plan).passed


def test_invalid_add_shape_mismatch():
    node = parse("add(x, y)")
    plan = _plan(node, "generic_expert", {"x": (32,), "y": (64,)})
    result = Verifier().check(node, plan)
    assert not result.passed
    assert result.level == 0


def test_valid_affine_form():
    node = normalize(parse("affine(A, x, b)"))
    plan = _plan(node, "affine_expert")
    assert Verifier().check(node, plan).passed


def test_wrong_expert_fails():
    node = parse("grad(f, x)")
    plan = _plan(node, "matmul_expert")
    result = Verifier().check(node, plan)
    assert not result.passed
    assert result.level == 2


def test_generic_expert_always_valid():
    node = parse("matmul(A, x)")
    plan = _plan(node, "generic_expert")
    assert Verifier().check(node, plan).passed


def test_mixed_dtypes_passes_with_warning():
    node = parse("add(x, y)")
    plan = _plan(node, "generic_expert",
                 input_dtypes={"x": "float32", "y": "float64"})
    assert Verifier().check(node, plan).passed


def test_constraint_expert_valid():
    node = parse("constraint(leq(matmul(A, x), b))")
    plan = _plan(node, "constraint_expert")
    assert Verifier().check(node, plan).passed


# ── Recursive tree check tests (B2) ──────────────────────────────────────────

def test_check_tree_valid_affine():
    root = normalize(parse("affine(A, x, b)"))
    env = {"A": (32, 64), "x": (64,), "b": (32,)}
    result = Verifier().check_tree(root, env)
    assert result.passed


def test_check_tree_invalid_matmul_inner_dim():
    # x has wrong shape: A is (32,64) but x is (32,) — inner dim mismatch
    root = normalize(parse("affine(A, x, b)"))
    env = {"A": (32, 64), "x": (32,), "b": (32,)}
    result = Verifier().check_tree(root, env)
    assert not result.passed


def test_check_tree_valid_matmul():
    root = parse("matmul(A, x)")
    env = {"A": (16, 32), "x": (32,)}
    result = Verifier().check_tree(root, env)
    assert result.passed


def test_check_tree_no_env_passes():
    # Without shape env, no shapes to check — should pass permissively
    root = normalize(parse("affine(A, x, b)"))
    result = Verifier().check_tree(root, {})
    assert result.passed
