from . import _backend
from .array import Array

def _call_op(a: Array, b: Array, op_type: _backend.ArrayOperationType) -> Array:
    """Handles shape checking and delegates to the unified C++ backend."""
    if a.shape != b.shape:
        raise ValueError(f"{op_type.name.lower()}: array shapes do not match")
    
    # Call the new unified C++ function
    h = _backend.operations_arrays(a._handle, b._handle, op_type)
    return Array.from_handle(h)

def add(a: Array, b: Array) -> Array:
    return _call_op(a, b, _backend.ArrayOperationType.ADD)

def sub(a: Array, b: Array) -> Array:
    return _call_op(a, b, _backend.ArrayOperationType.SUB)

def mul(a: Array, b: Array) -> Array:
    return _call_op(a, b, _backend.ArrayOperationType.MULT)

def div(a: Array, b: Array) -> Array:
    return _call_op(a, b, _backend.ArrayOperationType.DIV)

def array_add(self, other):
    if not isinstance(other, Array):
        return NotImplemented
    return add(self, other)

def array_radd(self, other):
    return array_add(self, other)

def array_sub(self, other):
    if not isinstance(other, Array):
        return NotImplemented
    return sub(self, other)

def array_mul(self, other):
    if not isinstance(other, Array):
        return NotImplemented
    return mul(self, other)

def array_radd(self, other):
    return array_add(self, other)

def array_div(self, other):
    if not isinstance(other, Array):
        return NotImplemented
    return div(self, other)

Array.__add__ = array_add
Array.__radd__ = array_radd
Array.__sub__ = array_sub
Array.__mul__ = array_mul
Array.__truediv__ = array_div