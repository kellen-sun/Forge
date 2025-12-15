from . import _backend
from .array import Array


def _call_op(a: Array, b: Array, op_type: str) -> Array:
    if a.shape != b.shape:
        raise ValueError(op_type + ": array shapes do not match")
    if op_type == "add":
        h = _backend.add(a._handle, b._handle)
    elif op_type == "sub":
        h = _backend.sub(a._handle, b._handle)
    elif op_type == "mul":
        h = _backend.mul(a._handle, b._handle)
    elif op_type == "div":
        h = _backend.div(a._handle, b._handle)
    else:
        raise ValueError("Unsupported operation type: " + op_type)
    return Array.from_handle(h)


def array_add(self, other):
    if other == 0:
        return self
    if not isinstance(other, Array):
        return NotImplemented
    return _call_op(self, other, "add")


def array_sub(self, other):
    if not isinstance(other, Array):
        return NotImplemented
    return _call_op(self, other, "sub")


def array_mul(self, other):
    if not isinstance(other, Array):
        return NotImplemented
    return _call_op(self, other, "mul")


def array_div(self, other):
    if not isinstance(other, Array):
        return NotImplemented
    return _call_op(self, other, "div")


Array.__add__ = array_add
Array.__radd__ = array_add
Array.__sub__ = array_sub
Array.__rsub__ = array_sub
Array.__mul__ = array_mul
Array.__rmul__ = array_mul
Array.__truediv__ = array_div
Array.__rtruediv__ = array_div
