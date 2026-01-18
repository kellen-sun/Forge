from typing import Sequence, Union

from . import graph
from .graph import Node, Ops
from .shape import _deduce_new_shape, _transpose_helper
from .utils import _default_strides, _indexing_helper


def _broadcast_shapes(s1, s2):
    if s1 == s2:
        return s1
    l1, l2 = len(s1), len(s2)
    length = max(l1, l2)
    s1 = (1,) * (length - l1) + s1
    s2 = (1,) * (length - l2) + s2
    out_shape = []
    for d1, d2 in zip(s1, s2):
        if d1 != d2 and d1 != 1 and d2 != 1:
            raise ValueError(f"broadcast_shapes: shape mismatch: {s1} vs {s2}")
        out_shape.append(max(d1, d2))
    return tuple(out_shape)


class SymbolicArray:
    def __init__(self, node: Node):
        self.node = node
        self.shape = node.shape
        self.offset = node.offset
        self.strides = node.strides

    def __add__(self, other):
        return self._binary_op(Ops.ADD, other)

    def __sub__(self, other):
        return self._binary_op(Ops.SUB, other)

    def __mul__(self, other):
        return self._binary_op(Ops.MUL, other)

    def __truediv__(self, other):
        return self._binary_op(Ops.DIV, other)

    def __matmul__(self, other):
        ndim_a = len(self.shape)
        ndim_b = len(other.shape)
        shape_a = self.shape
        shape_b = other.shape
        if ndim_a == 1:
            shape_a = (1,) + shape_a
        if ndim_b == 1:
            shape_b = shape_b + (1,)
        if shape_a[-1] != shape_b[-2]:
            raise ValueError(f"matmul: dimension mismatch {self.shape} @ {other.shape}")
        out_shape = list(
            _broadcast_shapes(shape_a[:-2], shape_b[:-2]) + (shape_a[-2], shape_b[-1])
        )
        if ndim_b == 1:
            out_shape.pop()
        if ndim_a == 1:
            col_idx = -2 if ndim_b != 1 else -1
            if len(out_shape) > 0:
                out_shape.pop(col_idx)
        out_shape = tuple(out_shape)
        new_node = Node(
            Ops.MATMUL,
            [self.node, other.node],
            out_shape,
            0,
            _default_strides(out_shape),
        )
        if graph.CURRENT_GRAPH:
            graph.CURRENT_GRAPH.add(new_node)
        return SymbolicArray(new_node)

    def _binary_op(self, op_code, other):
        new_node = Node(
            op_code,
            [self.node, other.node],
            self.shape,
            0,
            _default_strides(self.shape),
        )
        if graph.CURRENT_GRAPH:
            graph.CURRENT_GRAPH.add(new_node)
        return SymbolicArray(new_node)

    def __getitem__(self, key):
        new_shape, new_strides, new_offset = _indexing_helper(self, key)
        # TODO

    def __setitem__(self, key, value):
        new_shape, new_strides, new_offset = _indexing_helper(self, key)
        # TODO

    def reshape(self, *shape: Union[int, Sequence[int]]):
        new_shape = _deduce_new_shape(self, *shape)
        # the new offset depends on if we make a copy
        # which depends on if the old array was contiguous
        # if contiguous, we can use the same offset, and just redo strides
        # if not, a new Array is created in _backend (array_reshape) so offset = 0
        contiguous = True
        z = 1
        for i in range(len(self.shape) - 1, -1, -1):
            if self.strides[i] != z:
                contiguous = False
                break
            z *= self.shape[i]
        new_offset = 0
        if contiguous:
            new_offset = self.offset
        new_node = Node(
            Ops.RESHAPE,
            [self.node],
            new_shape,
            new_offset,
            _default_strides(new_shape),
        )
        if graph.CURRENT_GRAPH:
            graph.CURRENT_GRAPH.add(new_node)
        return SymbolicArray(new_node)

    def transpose(self, axes: Sequence[int] = None):
        new_shape, new_strides = _transpose_helper(self, axes)
        new_node = Node(Ops.TRANSPOSE, [self.node], new_shape, self.offset, new_strides)
        if graph.CURRENT_GRAPH:
            graph.CURRENT_GRAPH.add(new_node)
        return SymbolicArray(new_node)

    @property
    def T(self):
        return self.transpose()
