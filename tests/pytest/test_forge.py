import Forge
import pytest
from Forge import Array, forge
from Forge.ops import UNARY_OPS


def test_forge_elementwise_chain():
    @forge
    def f(a, b, c):
        return (a + b) * c

    a = Array([[1.0, 2.0], [3.0, 4.0]])
    b = Array([[4.0, 5.0], [6.0, 7.0]])
    c = Array([[1.0, 0.5], [2.0, 1.0]])
    eager = (a + b) * c
    compiled = f(a, b, c)
    assert compiled.list() == eager.list()
    assert compiled.shape == eager.shape


def test_forge_broadcast():
    @forge
    def f(a, b):
        return a + b

    a = Array([[1.0, 2.0]])
    b = Array([[1.0, 2.0], [3.0, 4.0]])
    eager = a + b
    compiled = f(a, b)
    assert compiled.list() == eager.list()
    assert compiled.shape == eager.shape


def test_forge_view():
    @forge
    def f(x):
        return x[1:] * 2.0

    x = Array([1.0, 2.0, 3.0, 4.0])
    eager = x[1:] * 2.0
    compiled = f(x)
    assert compiled.list() == eager.list()
    assert compiled.shape == eager.shape


def test_forge_sum_global():
    @forge
    def f(x):
        return x.sum()

    x = Array([[1.0, 2.0], [3.0, 4.0]])
    eager = x.sum()
    compiled = f(x)
    assert compiled.list() == eager.list()
    assert compiled.shape == eager.shape


def test_forge_sum_axis_keepdims():
    @forge
    def f(x):
        return x.sum(axis=1, keepdims=True)

    x = Array([[1.0, 2.0], [3.0, 4.0]])
    eager = x.sum(axis=1, keepdims=True)
    compiled = f(x)
    assert compiled.list() == eager.list()
    assert compiled.shape == eager.shape


def test_forge_sum_axis():
    @forge
    def f(x):
        return x.sum(axis=0)

    x = Array([[1.0, 2.0], [3.0, 4.0]])
    eager = x.sum(axis=0)
    compiled = f(x)
    assert compiled.list() == eager.list()
    assert compiled.shape == eager.shape


def test_forge_add_constant():
    @forge
    def f(x):
        return x + 1.5

    x = Array([1.0, 2.0, 3.0])
    eager = x + 1.5
    compiled = f(x)
    assert compiled.list() == eager.list()
    assert compiled.shape == eager.shape


def test_forge_reverse_mul_and_neg():
    @forge
    def f(x):
        return 2.0 * (-x)

    x = Array([1.0, -2.0, 3.0])
    eager = 2.0 * (-x)
    compiled = f(x)
    assert compiled.list() == eager.list()
    assert compiled.shape == eager.shape


@pytest.mark.parametrize("op_name", UNARY_OPS)
def test_forge_unary_matches_eager(op_name):
    @forge
    def f(x):
        return getattr(x, op_name)()

    x = Array([[0.25, 0.5], [0.75, 0.9]])
    eager = getattr(x, op_name)()
    compiled = f(x)
    assert compiled.list() == eager.list()
    assert compiled.shape == eager.shape


def test_forge_exp_module_and_view():
    @forge
    def f(x):
        return Forge.exp(x[1:])

    x = Array([0.25, 0.5, 0.75, 0.9])
    eager = Forge.exp(x[1:])
    compiled = f(x)
    assert compiled.list() == eager.list()
    assert compiled.shape == eager.shape


def test_forge_softmax():
    @forge
    def softmax(x):
        e = x.exp()
        return e / e.sum(axis=1, keepdims=True)

    x = Array([[1.0, 2.0, 3.0], [0.5, 0.0, -1.0]])
    e = x.exp()
    eager = e / e.sum(axis=1, keepdims=True)
    compiled = softmax(x)
    assert compiled.list() == eager.list()
    assert compiled.shape == eager.shape
