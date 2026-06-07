import pytest
from src.parser import parse, ParseError


def test_parse_var():
    node = parse("x")
    assert node.op == "var"
    assert node.value == "x"


def test_parse_add():
    node = parse("add(x, y)")
    assert node.op == "add"
    assert len(node.args) == 2
    assert node.args[0].value == "x"
    assert node.args[1].value == "y"


def test_parse_matmul():
    node = parse("matmul(A, x)")
    assert node.op == "matmul"
    assert node.args[0].value == "A"
    assert node.args[1].value == "x"


def test_parse_affine():
    node = parse("affine(A, x, b)")
    assert node.op == "affine"
    assert len(node.args) == 3


def test_parse_nested():
    node = parse("add(matmul(A, x), b)")
    assert node.op == "add"
    assert node.args[0].op == "matmul"
    assert node.args[1].value == "b"


def test_parse_grad():
    node = parse("grad(f, x)")
    assert node.op == "grad"


def test_parse_sum():
    node = parse("sum(i, x_i)")
    assert node.op == "sum"


def test_parse_constraint():
    node = parse("constraint(leq(matmul(A, x), b))")
    assert node.op == "constraint"
    assert node.args[0].op == "leq"
    assert node.args[0].args[0].op == "matmul"


def test_parse_whitespace_tolerance():
    node = parse("  add( x , y )  ")
    assert node.op == "add"
    assert len(node.args) == 2


def test_parse_invalid_raises():
    with pytest.raises(ParseError):
        parse("(")


def test_parse_repr_roundtrip():
    text = "add(matmul(A, x), b)"
    node = parse(text)
    assert repr(node) == text


# ── Numeric constant tests (A1) ───────────────────────────────────────────────

def test_parse_int_zero():
    node = parse("0")
    assert node.op == "const"
    assert node.value == 0


def test_parse_int_one():
    node = parse("1")
    assert node.op == "const"
    assert node.value == 1


def test_parse_int_positive():
    node = parse("42")
    assert node.op == "const"
    assert node.value == 42


def test_parse_float():
    node = parse("3.14")
    assert node.op == "const"
    assert abs(node.value - 3.14) < 1e-9


def test_parse_negative_int():
    node = parse("-2")
    assert node.op == "const"
    assert node.value == -2


def test_parse_negative_float():
    node = parse("-0.5")
    assert node.op == "const"
    assert abs(node.value - (-0.5)) < 1e-9


def test_parse_const_call_int():
    node = parse("const(0)")
    assert node.op == "const"
    assert node.value == 0


def test_parse_const_call_float():
    node = parse("const(1.5)")
    assert node.op == "const"
    assert abs(node.value - 1.5) < 1e-9


def test_parse_const_call_negative():
    node = parse("const(-3)")
    assert node.op == "const"
    assert node.value == -3


def test_parse_numeric_in_expression():
    node = parse("add(x, 0)")
    assert node.op == "add"
    assert node.args[1].op == "const"
    assert node.args[1].value == 0


def test_parse_mul_with_one():
    node = parse("mul(x, 1)")
    assert node.op == "mul"
    assert node.args[1].op == "const"
    assert node.args[1].value == 1
