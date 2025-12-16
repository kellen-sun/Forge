from typing import Sequence

from . import _backend
from .array import Array


def reshape(self, shape: Sequence[int]):
    pass


def transpose(self, axes: Sequence[int] = None):
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
    return Array(_backend.make_view(self._handle, new_shape, new_strides, self.offset))


@property
def T(self):
    return self.transpose()


Array.reshape = reshape
Array.transpose = transpose
Array.T = T
