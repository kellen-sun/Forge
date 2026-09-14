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


def test_forge_matmul_2d():
    @forge
    def f(a, b):
        return a @ b

    a = Array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = Array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    eager = a @ b
    compiled = f(a, b)
    assert compiled.list() == eager.list()
    assert compiled.shape == eager.shape


def test_forge_matmul_2d_transposed_input():
    @forge
    def f(a, b):
        return a @ b

    a = Array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]).T
    b = Array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    eager = a @ b
    compiled = f(a, b)
    assert compiled.list() == eager.list()
    assert compiled.shape == eager.shape


@pytest.mark.parametrize("case", ["matvec", "vecmat", "vecvec"])
def test_forge_matmul_vectors(case):
    @forge
    def f(a, b):
        return a @ b

    if case == "matvec":
        a = Array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = Array([7.0, 8.0, 9.0])
    elif case == "vecmat":
        a = Array([1.0, 2.0, 3.0])
        b = Array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    else:
        a = Array([1.0, 2.0, 3.0])
        b = Array([7.0, 9.0, 11.0])

    eager = a @ b
    compiled = f(a, b)
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


def test_forge_zeros():
    @forge
    def f():
        return Forge.zeros(2, 3)

    eager = Forge.zeros(2, 3)
    compiled = f()
    assert compiled.list() == eager.list()
    assert compiled.shape == eager.shape


def test_forge_zeros_in_expression():
    @forge
    def f(x):
        return x + Forge.zeros(x.shape)

    x = Array([[1.0, 2.0], [3.0, 4.0]])
    eager = x + Forge.zeros(x.shape)
    compiled = f(x)
    assert compiled.list() == eager.list()
    assert compiled.shape == eager.shape


@pytest.mark.parametrize("op_name", ["rand", "randn"])
def test_forge_random_factory_seed_progression(op_name):
    factory = getattr(Forge, op_name)

    @forge
    def f():
        return factory(2, 3)

    Forge.set_seed(42)
    eager = factory(2, 3)
    Forge.set_seed(42)
    compiled = f()
    assert compiled.list() == eager.list()
    assert compiled.shape == eager.shape

    next_compiled = f()
    assert next_compiled.list() != compiled.list()

    Forge.set_seed(42)
    replay = f()
    assert replay.list() == compiled.list()


def test_forge_update_scalar_view():
    @forge
    def f(x):
        x[1:] = 5.0
        return x

    x = Array([1.0, 2.0, 3.0, 4.0])
    result = f(x)
    assert result.list() == [1.0, 5.0, 5.0, 5.0]
    assert x.list() == [1.0, 5.0, 5.0, 5.0]


def test_forge_update_strided_view():
    @forge
    def f(x, value):
        x[::2] = value
        return x

    x = Array([0.0, 1.0, 2.0, 3.0])
    value = Array([10.0, 20.0])
    result = f(x, value)
    assert result.list() == [10.0, 1.0, 20.0, 3.0]


def test_forge_update_input_side_effect_when_not_returned():
    @forge
    def f(x, y):
        x[0] = 7.0
        return y

    x = Array([1.0, 2.0])
    y = Array([9.0])
    result = f(x, y)
    assert result.list() == [9.0]
    assert x.list() == [7.0, 2.0]


def test_forge_updates_execute_in_order():
    @forge
    def f(x):
        x[:] = 3.0
        x[1] = 4.0
        return x

    x = Array([0.0, 0.0, 0.0])
    assert f(x).list() == [3.0, 4.0, 3.0]


def test_forge_update_rejects_overlapping_rhs():
    @forge
    def f(x):
        x[1:] = x[:-1]
        return x

    with pytest.raises(RuntimeError, match="overlapping"):
        f(Array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize(
    ("op_name", "expected"),
    [
        ("iadd", [4.0, 6.0]),
        ("isub", [-2.0, -2.0]),
        ("imul", [3.0, 8.0]),
        ("idiv", [1.0 / 3.0, 0.5]),
    ],
)
def test_forge_arithmetic_update(op_name, expected):
    @forge
    def f(x, value):
        if op_name == "iadd":
            x += value
        elif op_name == "isub":
            x -= value
        elif op_name == "imul":
            x *= value
        else:
            x /= value
        return x

    x = Array([1.0, 2.0])
    value = Array([3.0, 4.0])
    result = f(x, value)
    assert result.list() == pytest.approx(expected, rel=1e-6)


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
