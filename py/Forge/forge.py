"""
JIT compilation decorator using tracing.

The @forge decorator traces Python function execution on first call
and replays the captured operation graph on subsequent calls.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from . import _backend
from .array import Array
from .tracer import OpNode, TracedArray, TracedGraph, Tracer


class TracedFunction:
    """
    Executes a traced computation graph.

    This class wraps a TracedGraph and provides execution via eager ops.
    """

    def __init__(self, graph: TracedGraph):
        self._graph = graph

    def __call__(self, *args: Array) -> Union[Array, Tuple[Array, ...]]:
        """Execute the traced graph with the given Array inputs."""
        # Validate input count
        if len(args) != len(self._graph.input_indices):
            raise ValueError(
                f"Expected {len(self._graph.input_indices)} inputs, "
                f"got {len(args)}"
            )

        # Validate input shapes
        for i, (arg, input_idx) in enumerate(zip(args, self._graph.input_indices)):
            expected_shape = self._graph.shapes[input_idx]
            if arg.shape != expected_shape:
                raise ValueError(
                    f"Input {i} shape mismatch: expected {expected_shape}, "
                    f"got {arg.shape}"
                )

        # Build value storage - maps value indices to Arrays
        values: Dict[int, Array] = {}

        # Store inputs
        for arg, input_idx in zip(args, self._graph.input_indices):
            values[input_idx] = arg

        # Execute operations in order
        for node in self._graph.nodes:
            inputs = [values[idx] for idx in node.inputs]
            result = self._execute_op(node.op_type, inputs)
            values[node.output_idx] = result

        # Gather outputs
        outputs = [values[idx] for idx in self._graph.output_indices]

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def _execute_op(self, op_type: str, inputs: List[Array]) -> Array:
        """Execute a single operation using eager backend ops."""
        if op_type == "add":
            return Array.from_handle(_backend.add(inputs[0]._handle, inputs[1]._handle))
        elif op_type == "sub":
            return Array.from_handle(_backend.sub(inputs[0]._handle, inputs[1]._handle))
        elif op_type == "mul":
            return Array.from_handle(_backend.mul(inputs[0]._handle, inputs[1]._handle))
        elif op_type == "div":
            return Array.from_handle(_backend.div(inputs[0]._handle, inputs[1]._handle))
        elif op_type == "matmul":
            return Array.from_handle(_backend.matmul(inputs[0]._handle, inputs[1]._handle))
        else:
            raise ValueError(f"Unknown operation type: {op_type}")


class ForgeFunction:
    """
    Wrapper returned by the @forge decorator.

    On first call, traces the wrapped function and caches the TracedFunction.
    On subsequent calls, replays the cached graph.
    """

    def __init__(self, fn: Callable):
        self._fn = fn
        self._cached_trace: Optional[TracedFunction] = None
        self._traced_shapes: Optional[Tuple[Tuple[int, ...], ...]] = None
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Array) -> Union[Array, Tuple[Array, ...]]:
        """Execute the function, tracing on first call."""
        # Get input shapes
        input_shapes = tuple(arg.shape for arg in args)

        # Check if we need to trace (first call or shape mismatch)
        if self._cached_trace is None or self._traced_shapes != input_shapes:
            self._trace(args)

        # Execute cached trace
        return self._cached_trace(*args)

    def _trace(self, args: Tuple[Array, ...]) -> None:
        """Trace the function with the given argument shapes."""
        tracer = Tracer()

        # Create TracedArray inputs with matching shapes
        traced_inputs = [tracer.create_input(arg.shape) for arg in args]

        # Execute function with traced inputs
        result = self._fn(*traced_inputs)

        # Handle single or multiple outputs
        if isinstance(result, TracedArray):
            outputs = [result]
        elif isinstance(result, (list, tuple)):
            outputs = list(result)
        else:
            raise TypeError(
                f"Function must return TracedArray or tuple of TracedArray, "
                f"got {type(result)}"
            )

        # Build and cache the graph
        graph = tracer.build_graph(outputs)
        self._cached_trace = TracedFunction(graph)
        self._traced_shapes = tuple(arg.shape for arg in args)

    def _get_graph(self) -> Optional[TracedGraph]:
        """Return the cached TracedGraph for debugging."""
        if self._cached_trace is None:
            return None
        return self._cached_trace._graph

    def _clear_cache(self) -> None:
        """Clear the cached trace (useful for testing)."""
        self._cached_trace = None
        self._traced_shapes = None


def forge(fn: Callable) -> ForgeFunction:
    """
    Decorator that enables tracing-based JIT compilation.

    Usage:
        @forge
        def my_fn(a, b):
            return a + b

        # First call: traces function, caches TracedFunction
        result = my_fn(Array([1, 2]), Array([3, 4]))

        # Subsequent calls: replays cached graph
        result = my_fn(Array([5, 6]), Array([7, 8]))

    The traced graph can be inspected via:
        my_fn._get_graph()
    """
    return ForgeFunction(fn)
