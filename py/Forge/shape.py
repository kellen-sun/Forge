from typing import Sequence, Union

from . import _backend
from .array import Array
from .utils import _deduce_new_shape


def reshape(self, *shape: Union[int, Sequence[int]]):
    new_shape = _deduce_new_shape(self, *shape)
    return Array(_backend.reshape(self._handle, new_shape))


def _transpose_helper(self, axes: Sequence[int] = None):
    if axes is None:
        axes = range(len(self.shape) - 1, -1, -1)
    ndim = len(self.shape)
    if len(axes) != ndim:
        raise ValueError(
            f"Array: Transpose, axes don't match array: axes={len(axes)}, array={ndim}"
        )
    if set(axes) != set(range(ndim)):
        raise ValueError("Array: Tranpose, axes must be a permutation of dimensions")
    new_shape = [self.shape[i] for i in axes]
    new_strides = [self.strides[i] for i in axes]
    return new_shape, new_strides


def transpose(self, axes: Sequence[int] = None):
    new_shape, new_strides = _transpose_helper(self, axes)
    return Array(_backend.make_view(self._handle, new_shape, new_strides, self.offset))


@property
def T(self):
    return self.transpose()


Array.reshape = reshape
Array.transpose = transpose
Array.T = T
