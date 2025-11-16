from . import _backend
from .array import Array

def add(a: Array, b: Array) -> Array:
    if a.shape != b.shape:
        raise ValueError("add: array shapes do not match")
    h = _backend.add_arrays(a._handle, b._handle)
    return Array.from_handle(h)

def array_add(self, other):
    if not isinstance(other, Array):
        return NotImplemented
    return add(self, other)

def array_radd(self, other):
    return array_add(self, other)

Array.__add__ = array_add
Array.__radd__ = array_radd
