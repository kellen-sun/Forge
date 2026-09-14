from typing import Sequence, Union

from . import _backend
from .array import Array
from . import graph
from .graph import Node, Ops
from .symbolic import SymbolicArray
from .utils import _default_strides, _normalize_sum_axis


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


# Order is OpCode.UNARY args[0]; keep in sync with kUnaryNames in cpp/include/common.h
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


for kind, op_name in enumerate(UNARY_OPS):
    backend_fn = getattr(_backend, op_name)

    def unary_wrapper(x, _fn=backend_fn, _kind=kind):
        if isinstance(x, SymbolicArray):
            return x._unary(_kind)
        return Array.from_handle(_fn(x._handle))

    unary_wrapper.__name__ = op_name
    globals()[op_name] = unary_wrapper
    setattr(Array, op_name, unary_wrapper)
    setattr(SymbolicArray, op_name, unary_wrapper)


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
        if op_name == "zeros" and graph.CURRENT_GRAPH is not None:
            shape = tuple(shape)
            node = Node(
                Ops.ZEROS,
                [],
                shape,
                0,
                _default_strides(shape),
            )
            graph.CURRENT_GRAPH.add(node)
            return SymbolicArray(node)
        return Array.from_handle(_fn(shape))

    nullary_wrapper.__name__ = op_name
    globals()[op_name] = nullary_wrapper


def sum(self, axis=None, keepdims=False):
    if axis is None:
        h = _backend.sum_global(self._handle, keepdims)
        out_array = Array.from_handle(h)
        return out_array

    axis = _normalize_sum_axis(self.shape, axis)
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
