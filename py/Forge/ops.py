from typing import Sequence, Union

from . import _backend
from .array import Array


def _call_op(a: Array, b: Array, op_type: str) -> Array:
    if isinstance(a, (list, tuple)):
        a = Array(a)
    if isinstance(a, (int, float)):
        a = Array([a])
    if not isinstance(a, Array):
        return NotImplemented
    if isinstance(b, (list, tuple)):
        b = Array(b)
    if isinstance(b, (int, float)):
        b = Array([b])
    if not isinstance(b, Array):
        return NotImplemented
    if op_type == "add":
        h = _backend.add(a._handle, b._handle)
    elif op_type == "sub":
        h = _backend.sub(a._handle, b._handle)
    elif op_type == "mul":
        h = _backend.mul(a._handle, b._handle)
    elif op_type == "div":
        h = _backend.div(a._handle, b._handle)
    elif op_type == "iadd":
        h = _backend.iadd(a._handle, b._handle)
    elif op_type == "isub":
        h = _backend.isub(a._handle, b._handle)
    elif op_type == "imul":
        h = _backend.imul(a._handle, b._handle)
    elif op_type == "idiv":
        h = _backend.idiv(a._handle, b._handle)
    else:
        raise ValueError("Unsupported operation type: " + op_type)
    return Array.from_handle(h)


def array_add(self, other):
    return _call_op(self, other, "add")


def array_sub(self, other):
    return _call_op(self, other, "sub")


def array_neg(self):
    return _call_op(0, self, "sub")


def array_mul(self, other):
    return _call_op(self, other, "mul")


def array_div(self, other):
    return _call_op(self, other, "div")


def array_matmul(self, other):
    if not isinstance(other, Array):
        return NotImplemented
    return Array(_backend.matmul(self._handle, other._handle))


def array_iadd(self, other):
    return _call_op(self, other, "iadd")


def array_isub(self, other):
    return _call_op(self, other, "isub")


def array_imul(self, other):
    return _call_op(self, other, "imul")


def array_idiv(self, other):
    return _call_op(self, other, "idiv")


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


Array.__add__ = array_add
Array.__radd__ = array_add
Array.__pos__ = lambda x: x
Array.__sub__ = array_sub
Array.__rsub__ = array_sub
Array.__neg__ = array_neg
Array.__mul__ = array_mul
Array.__rmul__ = array_mul
Array.__truediv__ = array_div
Array.__rtruediv__ = array_div
Array.__matmul__ = array_matmul
Array.__iadd__ = array_iadd
Array.__isub__ = array_isub
Array.__imul__ = array_imul
Array.__itruediv__ = array_idiv
Array.sum = sum
