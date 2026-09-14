from Forge import Array, forge


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


def test_forge_add_constant():
    @forge
    def f(x):
        return x + 1.5

    x = Array([1.0, 2.0, 3.0])
    eager = x + 1.5
    compiled = f(x)
    assert compiled.list() == eager.list()
    assert compiled.shape == eager.shape
