from array import array
from typing import Sequence
from . import _backend
import numpy as np

def _infer_shape_and_flatten(x):
    """
    Takes nested lists/tuples and returns (shape, flat_list_of_floats).
    ValueError if missing values. Only floats for now.
    """
    if isinstance(x, (int, float)):
        return (), [float(x)]
    if isinstance(x, (list, tuple)):
        # Recurse
        if len(x) == 0:
            return (0,), []
        shapes = []
        flat = []
        for el in x:
            s, f = _infer_shape_and_flatten(el)
            shapes.append(s)
            flat.extend(f)
        first = shapes[0]
        for s in shapes:
            if s != first:
                raise ValueError("ragged nested lists: differing inner shapes")
        return (len(x), ) + first, flat
    if isinstance(x, (bytes, bytearray, memoryview)):
        raise TypeError("shape required for raw bytes; use Array.from_buffer(buf, shape) and specify shape")
    if isinstance(x, array):
        if x.typecode != "f":
            raise TypeError("array must be of type 'float32'")
        return (len(x),), list(x)
    raise TypeError(f"unsupported input type: {type(x)}")

class Array:
    """
    Python Array that the library provides. 
    Stores only metadata and a backend handle (where the data is).
    """

    def __init__(self, data):
        """
        Accepts:
         - Existing Array
         - Nested Python lists/tuples
         - Python array('f') type
         - bytes/memoryview with Array.from_buffer
        """
        if isinstance(data, Array):
            # Existing Array
            self._handle = data._handle
            self.shape = data.shape
            return
        
        try:
            backend_type = _backend.ArrayHandle
        except Exception:
            backend_type = None
        if backend_type is not None and isinstance(data, backend_type):
            # Passed a backend array handle
            self._handle = data
            self.shape = tuple(_backend.array_shape(self._handle))
            return
        
        if isinstance(data, array):
            # Python array('f') type
            if data.typecode != "f":
                raise TypeError("array must have type float32")
            shape = (len(data),)
            mv = memoryview(data)
            self._handle = _backend.create_array_from_buffer(mv, list(shape))
            self.shape = shape
            return
        
        else:
            # Nested Python lists/tuples
            shape, flat = _infer_shape_and_flatten(data)
            buf = array('f', flat)
            mv = memoryview(buf)
            self._handle = _backend.create_array_from_buffer(mv, list(shape))
            self.shape = shape
            return
    
    @classmethod
    def from_buffer(cls, buf, shape: Sequence[int]):
        """Construct Array from memoryview/array('f') with explicit shape"""
        mv = memoryview(buf)
        inst = cls.__new__(cls)
        inst._handle = _backend.create_array_from_buffer(mv, list(shape))
        inst.shape = tuple(shape)
        return inst

    @classmethod
    def from_handle(cls, handle):
        """Construct Array from a backend array handle"""
        inst = cls.__new__(cls)
        inst._handle = handle
        inst.shape = tuple(_backend.array_shape(handle))
        return inst
    
    @property
    def strides(self):
        return self._handle.strides

    @property
    def offset(self):
        return self._handle.offset
    
    def list(self):
        """Return back a nested list form"""
        return _backend.array_to_list(self._handle)
    
    def __repr__(self):
        return f"Array(shape = {self.shape})"
    
    def __getitem__(self, key):
        """
        Handles slicing: a[1], a[1:5], a[::2]
        Does NOT copy data. Returns a View.
        """
        if not isinstance(key, tuple):
            key = (key,)

        # 1. Start with current metadata
        new_shape = list(self.shape)
        new_strides = list(self.strides)
        new_offset = self.offset

        # 2. Iterate over dims
        # (Simplified logic - assumes simple slicing for now)
        dim = 0
        for s in key:
            if isinstance(s, int):
                # Integer Index: Reduce dim
                # Offset increases by index * stride
                new_offset += s * new_strides[dim]
                new_shape.pop(dim)
                new_strides.pop(dim)
                # Don't increment dim, as we popped it
                
            elif isinstance(s, slice):
                # Slice: Adjust shape and stride
                start, stop, step = s.indices(new_shape[dim])
                
                # Update Offset (start point)
                new_offset += start * new_strides[dim]
                
                # Update Stride (step)
                new_strides[dim] *= step
                
                # Update Shape (length)
                new_shape[dim] = (stop - start + (step - 1)) // step
                dim += 1
                
            else:
                raise TypeError("Only int and slice supported")

        # 3. Call C++ to create the view handle
        
        handle = _backend.make_view(self._handle, new_shape, new_strides, new_offset)
        if len(new_shape) == 0:
            return handle.item()
        return Array(handle)
