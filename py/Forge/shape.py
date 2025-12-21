from typing import Sequence, Union

from . import _backend
from .array import Array


def reshape(self, *shape: Union[int, Sequence[int]]):
    if len(shape) == 1:
        arg = shape[0]
        if isinstance(arg, int):
            shape = (arg,)
        else:
            shape = list(arg)
    numel = 1
    for s in self.shape:
        numel *= s
    if -1 in shape:
        if shape.count(-1) > 1:
            raise ValueError("Array: reshape can only infer one dimension")
        other_size = 1
        for s in shape:
            other_size *= s
        if other_size == 0 or numel % -other_size != 0:
            raise ValueError(
                f"Array: cannot reshape array of size {numel} into shape {shape}"
            )
        inferred = numel // -other_size
        new_shape = [inferred if s == -1 else s for s in shape]
    else:
        new_size = 1
        for s in shape:
            new_size *= s
        if new_size != numel:
            raise ValueError(
                f"Array: cannot reshape array of size {numel} into shape {shape}"
            )
        new_shape = list(shape)
    return Array(_backend.reshape(self._handle, new_shape))


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
