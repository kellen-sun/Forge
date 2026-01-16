"""
Tracing infrastructure for the @forge JIT decorator.

This module provides the core classes for tracing Python function execution
and building a computation graph that can be replayed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

from . import _backend
from .array import Array


@dataclass
class OpNode:
    """Single operation record in the traced graph."""

    op_type: str  # "add", "sub", "mul", "div", "matmul"
    inputs: Tuple[int, ...]  # Indices into value list
    output_idx: int  # Output index in values list
    shape: Tuple[int, ...]  # Output shape


@dataclass
class TracedGraph:
    """Complete computation graph from tracing."""

    input_indices: List[int]  # Indices of input values
    output_indices: List[int]  # Indices of output values
    nodes: List[OpNode]  # Operations in execution order
    shapes: List[Tuple[int, ...]]  # Shapes for all values

    def __repr__(self) -> str:
        lines = ["TracedGraph("]
        lines.append(f"  inputs: {self.input_indices}")
        lines.append(f"  outputs: {self.output_indices}")
        lines.append(f"  nodes: [")
        for node in self.nodes:
            lines.append(f"    {node.op_type}(inputs={node.inputs}) -> {node.output_idx} (shape={node.shape})")
        lines.append("  ]")
        lines.append(")")
        return "\n".join(lines)


class TracedArray:
    """
    Proxy array that records operations instead of executing them.

    Each TracedArray is associated with a Tracer and has an index
    into the tracer's value list.
    """

    def __init__(self, tracer: Tracer, idx: int, shape: Tuple[int, ...]):
        self._tracer = tracer
        self._idx = idx
        self._shape = shape

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    def __add__(self, other: TracedArray) -> TracedArray:
        if not isinstance(other, TracedArray):
            return NotImplemented
        return self._tracer._record_op("add", self, other)

    def __radd__(self, other: TracedArray) -> TracedArray:
        if not isinstance(other, TracedArray):
            return NotImplemented
        return self._tracer._record_op("add", other, self)

    def __sub__(self, other: TracedArray) -> TracedArray:
        if not isinstance(other, TracedArray):
            return NotImplemented
        return self._tracer._record_op("sub", self, other)

    def __rsub__(self, other: TracedArray) -> TracedArray:
        if not isinstance(other, TracedArray):
            return NotImplemented
        return self._tracer._record_op("sub", other, self)

    def __mul__(self, other: TracedArray) -> TracedArray:
        if not isinstance(other, TracedArray):
            return NotImplemented
        return self._tracer._record_op("mul", self, other)

    def __rmul__(self, other: TracedArray) -> TracedArray:
        if not isinstance(other, TracedArray):
            return NotImplemented
        return self._tracer._record_op("mul", other, self)

    def __truediv__(self, other: TracedArray) -> TracedArray:
        if not isinstance(other, TracedArray):
            return NotImplemented
        return self._tracer._record_op("div", self, other)

    def __rtruediv__(self, other: TracedArray) -> TracedArray:
        if not isinstance(other, TracedArray):
            return NotImplemented
        return self._tracer._record_op("div", other, self)

    def __matmul__(self, other: TracedArray) -> TracedArray:
        if not isinstance(other, TracedArray):
            return NotImplemented
        return self._tracer._record_op("matmul", self, other)

    def __repr__(self) -> str:
        return f"TracedArray(idx={self._idx}, shape={self._shape})"


def _compute_matmul_shape(
    shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Compute the output shape for matmul of two arrays."""
    # Handle 1D cases
    if len(shape_a) == 1 and len(shape_b) == 1:
        # vec @ vec -> scalar
        return ()
    elif len(shape_a) == 1:
        # vec @ mat -> vec
        return shape_b[:-2] + (shape_b[-1],) if len(shape_b) > 1 else (shape_b[-1],)
    elif len(shape_b) == 1:
        # mat @ vec -> vec
        return shape_a[:-1]
    else:
        # mat @ mat -> mat (with broadcasting for batch dims)
        batch_a = shape_a[:-2]
        batch_b = shape_b[:-2]

        # Broadcast batch dimensions
        max_batch_len = max(len(batch_a), len(batch_b))
        batch_a = (1,) * (max_batch_len - len(batch_a)) + batch_a
        batch_b = (1,) * (max_batch_len - len(batch_b)) + batch_b

        batch_out = []
        for a, b in zip(batch_a, batch_b):
            batch_out.append(max(a, b))

        return tuple(batch_out) + (shape_a[-2], shape_b[-1])


class Tracer:
    """
    Context for building computation graphs by tracing operations.

    Usage:
        tracer = Tracer()
        a = tracer.create_input((2, 3))
        b = tracer.create_input((2, 3))
        c = a + b
        graph = tracer.build_graph([c])
    """

    def __init__(self):
        self._nodes: List[OpNode] = []
        self._shapes: List[Tuple[int, ...]] = []
        self._input_indices: List[int] = []
        self._next_idx = 0

    def create_input(self, shape: Tuple[int, ...]) -> TracedArray:
        """Create a TracedArray representing a function input."""
        idx = self._next_idx
        self._next_idx += 1
        self._shapes.append(shape)
        self._input_indices.append(idx)
        return TracedArray(self, idx, shape)

    def _record_op(
        self, op_type: str, *inputs: TracedArray
    ) -> TracedArray:
        """Record an operation and return a new TracedArray for its output."""
        input_indices = tuple(inp._idx for inp in inputs)

        # Compute output shape
        if op_type == "matmul":
            out_shape = _compute_matmul_shape(inputs[0]._shape, inputs[1]._shape)
        else:
            # Element-wise ops require same shape
            out_shape = inputs[0]._shape

        output_idx = self._next_idx
        self._next_idx += 1
        self._shapes.append(out_shape)

        node = OpNode(
            op_type=op_type,
            inputs=input_indices,
            output_idx=output_idx,
            shape=out_shape,
        )
        self._nodes.append(node)

        return TracedArray(self, output_idx, out_shape)

    def build_graph(self, outputs: List[TracedArray]) -> TracedGraph:
        """Build the final TracedGraph from the traced outputs."""
        output_indices = [out._idx for out in outputs]

        return TracedGraph(
            input_indices=list(self._input_indices),
            output_indices=output_indices,
            nodes=list(self._nodes),
            shapes=list(self._shapes),
        )
