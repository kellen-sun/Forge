from array import array
from typing import Sequence

from . import _backend


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
        return (len(x),) + first, flat
    if isinstance(x, (bytes, bytearray, memoryview)):
        raise TypeError(
            "shape required for raw bytes; use Array.from_buffer(buf, shape) and specify shape"
        )
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
            buf = array("f", flat)
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
        return f"Array(shape = {self.shape}, type=float)\n" + str(self.list())

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return self.shape[0]

    def _indexing_helper(self, key):
        if isinstance(key, list):
            raise TypeError(
                "Array: fancy indexing (passing lists/arrays as indices) is not supported"
            )
        if not isinstance(key, tuple):
            key = (key,)

        ellipsis_count = key.count(Ellipsis)
        if ellipsis_count > 1:
            raise IndexError("Array: only one ellipsis allowed in indexing")
        if ellipsis_count == 1:
            ellipsis_index = key.index(Ellipsis)
            explicit_count = sum(1 for k in key if k is not Ellipsis and k is not None)
            if len(self.shape) < explicit_count:
                raise IndexError("Array: too many indices for array")
            num_missing = len(self.shape) - explicit_count
            key = (
                key[:ellipsis_index]
                + (slice(None),) * num_missing
                + key[ellipsis_index + 1 :]
            )
            key = tuple(key)

        new_shape = list(self.shape)
        new_strides = list(self.strides)
        new_offset = self.offset

        dim = 0
        deleted = 0
        if len(self.shape) < len(key) - key.count(None):
            raise IndexError("Array: too many indices for array")
        for s in key:
            if s is None:
                new_shape.insert(dim, 1)
                new_strides.insert(dim, 0)
                deleted -= 1

            elif isinstance(s, int):
                if s < 0:
                    s += self.shape[dim]
                if s < 0 or s >= self.shape[dim]:
                    raise IndexError("Array: Index out of range")

                new_offset += s * new_strides[dim - deleted]
                new_shape.pop(dim - deleted)
                new_strides.pop(dim - deleted)
                deleted += 1
                dim += 1

            elif isinstance(s, slice):
                if s.step == 0:
                    raise ValueError("Array: slice step cannot be zero")
                if s.step is None:
                    step = 1
                else:
                    step = s.step
                if s.start is None:
                    if step > 0:
                        start = 0
                    else:
                        start = self.shape[dim] - 1
                else:
                    start = s.start
                if s.stop is None:
                    if step > 0:
                        stop = self.shape[dim]
                    elif self.shape[dim] == 0:
                        stop = 0
                    else:
                        stop = -1 - self.shape[dim]
                else:
                    stop = s.stop

                if start < 0:
                    start += self.shape[dim]
                if stop < 0:
                    stop += self.shape[dim]
                if start < 0:
                    start = 0
                if start >= self.shape[dim] and self.shape[dim]:
                    start = self.shape[dim] - 1
                if stop < -1:
                    stop = -1
                if stop > self.shape[dim]:
                    stop = self.shape[dim]

                new_offset += start * new_strides[dim - deleted]
                new_strides[dim - deleted] *= step
                sgn = 1 if step > 0 else -1
                new_shape[dim - deleted] = max(
                    0, (sgn * (stop - start) + abs(step) - 1) // abs(step)
                )
                dim += 1

            else:
                raise TypeError("Array: Only int and slice supported")

        return new_shape, new_strides, new_offset

    def __getitem__(self, key):
        """
        Indexing and slicing, returns a new Array that refers to the same data.
        Index by [3], [3,4], or negatives [-1] or slice [3:5].
        """
        new_shape, new_strides, new_offset = self._indexing_helper(key)

        handle = _backend.make_view(self._handle, new_shape, new_strides, new_offset)
        if len(new_shape) == 0:
            return handle.item()
        return Array(handle)

    def __setitem__(self, key, value):
        """
        Set values of an indexed into view.
        Value can be a scalar or an Array of matching shape.
        """
        new_shape, new_strides, new_offset = self._indexing_helper(key)

        if isinstance(value, (list, tuple)):
            value = Array(value)

        if isinstance(value, (int, float)):
            val_handle = Array([value])._handle
        elif isinstance(value, Array):
            size = 1
            for dim in value.shape:
                size *= dim
            if value.shape == tuple(new_shape) or size == 1:
                val_handle = value._handle
            else:
                raise ValueError("Array: assignment shape mismatch")
        else:
            raise TypeError(
                "Array: assignment value must be a scalar, \
                    or Array/nested lists/tuples of matching shape"
            )

        _backend.copy_to_view(
            self._handle, val_handle, new_shape, new_strides, new_offset
        )
