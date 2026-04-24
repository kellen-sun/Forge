from typing import Sequence, Union

from . import _backend
from .array import Array


def _to_array(x):
    if isinstance(x, Array):
        return x
    if isinstance(x, (list, tuple)):
        return Array(x)
    if isinstance(x, (int, float)):
        return Array([x])
    return NotImplemented


def _make_binop(op_name):
    backend_fn = getattr(_backend, op_name)

    def method(self, other):
        a, b = self, _to_array(other)
        if b is NotImplemented:
            return NotImplemented
        return Array.from_handle(backend_fn(a._handle, b._handle))

    return method


def _make_rbinop(op_name):
    backend_fn = getattr(_backend, op_name)

    def method(self, other):
        a, b = _to_array(other), self
        if a is NotImplemented:
            return NotImplemented
        return Array.from_handle(backend_fn(a._handle, b._handle))

    return method


def array_matmul(self, other):
    if not isinstance(other, Array):
        return NotImplemented
    return Array(_backend.matmul(self._handle, other._handle))


UNARY_OPS = [
    "exp",
    "exp2",
    "exp10",
    "log",
    "log2",
    "log10",
    "sqrt",
    "rsqrt",
    "abs",
    "sign",
    "ceil",
    "floor",
    "round",
    "trunc",
    "fract",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
]


for op_name in UNARY_OPS:
    backend_fn = getattr(_backend, op_name)

    def unary_wrapper(x: Array, _fn=backend_fn) -> Array:
        return Array.from_handle(_fn(x._handle))

    unary_wrapper.__name__ = op_name
    globals()[op_name] = unary_wrapper
    setattr(Array, op_name, unary_wrapper)


NULLARY_OPS = ["rand", "randn", "zeros"]


for op_name in NULLARY_OPS:
    backend_fn = getattr(_backend, op_name)

    def nullary_wrapper(*shape: Union[int, Sequence[int]], _fn=backend_fn) -> Array:
        if len(shape) == 1:
            arg = shape[0]
            if isinstance(arg, int):
                shape = [arg]
            else:
                shape = list(arg)
        else:
            shape = list(shape)
        return Array.from_handle(_fn(shape))

    nullary_wrapper.__name__ = op_name
    globals()[op_name] = nullary_wrapper


def sum(self, axis=None, keepdims=False):
    if axis is None:
        h = _backend.sum_global(self._handle, keepdims)
        out_array = Array.from_handle(h)
        return out_array

    if not isinstance(axis, int):
        raise TypeError("axis must be an integer or None")

    if axis < 0:
        axis += len(self.shape)
    if axis < 0 or axis >= len(self.shape):
        raise IndexError(
            f"Array: Axis {axis} is out of bounds for Array of dimension {len(self.shape)}"
        )

    h = _backend.sum_axis(self._handle, axis, keepdims)
    return Array.from_handle(h)


Array.__pos__ = lambda self: self
Array.__neg__ = lambda self: Array.from_handle(
    _backend.sub(_to_array(0)._handle, self._handle)
)
Array.__add__ = _make_binop("add")
Array.__radd__ = Array.__add__
Array.__sub__ = _make_binop("sub")
Array.__rsub__ = _make_rbinop("sub")
Array.__mul__ = _make_binop("mul")
Array.__rmul__ = Array.__mul__
Array.__truediv__ = _make_binop("div")
Array.__rtruediv__ = _make_rbinop("div")
Array.__iadd__ = _make_binop("iadd")
Array.__isub__ = _make_binop("isub")
Array.__imul__ = _make_binop("imul")
Array.__itruediv__ = _make_binop("idiv")
Array.__matmul__ = array_matmul
Array.sum = sum
