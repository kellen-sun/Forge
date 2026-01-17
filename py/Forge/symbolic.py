from . import graph
from .graph import Node, Ops


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

    def __add__(self, other):
        self._binary_op(Ops.ADD, other)

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
        new_node = Node(Ops.MATMUL, [self.node, other.node], out_shape)
        if graph.CURRENT_GRAPH:
            graph.CURRENT_GRAPH.add(new_node)
        return SymbolicArray(new_node)

    def _binary_op(self, op_code, other):
        new_node = Node(op_code, [self.node, other.node], self.shape)
        if graph.CURRENT_GRAPH:
            graph.CURRENT_GRAPH.add(new_node)
        return SymbolicArray(new_node)
