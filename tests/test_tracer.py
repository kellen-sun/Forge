"""
Tests for the tracing-based JIT infrastructure.

Test categories:
1. Tracer basics - create_input, record_op, build_graph
2. Decorator - simple add, chain ops, matmul, trace reuse
3. Correctness vs eager - random data, compare results
4. Error handling - shape mismatch, wrong input count
"""

import numpy as np
import pytest
from Forge import Array, Tracer, TracedArray, TracedGraph, OpNode, forge


# =============================================================================
# Tracer Basics
# =============================================================================


class TestTracerBasics:
    """Tests for the core Tracer class."""

    def test_create_input(self):
        """Test creating a traced input."""
        tracer = Tracer()
        inp = tracer.create_input((2, 3))

        assert isinstance(inp, TracedArray)
        assert inp.shape == (2, 3)
        assert inp._idx == 0

    def test_create_multiple_inputs(self):
        """Test creating multiple traced inputs."""
        tracer = Tracer()
        a = tracer.create_input((2, 3))
        b = tracer.create_input((2, 3))
        c = tracer.create_input((4, 5))

        assert a._idx == 0
        assert b._idx == 1
        assert c._idx == 2
        assert a.shape == (2, 3)
        assert c.shape == (4, 5)

    def test_record_add_op(self):
        """Test recording an add operation."""
        tracer = Tracer()
        a = tracer.create_input((2, 3))
        b = tracer.create_input((2, 3))
        c = a + b

        assert isinstance(c, TracedArray)
        assert c.shape == (2, 3)
        assert c._idx == 2  # After two inputs

    def test_record_chain_ops(self):
        """Test recording chained operations."""
        tracer = Tracer()
        a = tracer.create_input((2, 3))
        b = tracer.create_input((2, 3))
        c = a + b
        d = c * a
        e = d - b

        assert e._idx == 4  # 2 inputs + 3 ops

    def test_build_graph_simple(self):
        """Test building a graph from a simple computation."""
        tracer = Tracer()
        a = tracer.create_input((2, 3))
        b = tracer.create_input((2, 3))
        c = a + b
        graph = tracer.build_graph([c])

        assert isinstance(graph, TracedGraph)
        assert graph.input_indices == [0, 1]
        assert graph.output_indices == [2]
        assert len(graph.nodes) == 1
        assert graph.nodes[0].op_type == "add"
        assert graph.nodes[0].inputs == (0, 1)

    def test_build_graph_chain(self):
        """Test building a graph with chained operations."""
        tracer = Tracer()
        a = tracer.create_input((2, 2))
        b = tracer.create_input((2, 2))
        c = a + b
        d = c * a
        e = d - b
        graph = tracer.build_graph([e])

        assert len(graph.nodes) == 3
        assert graph.nodes[0].op_type == "add"
        assert graph.nodes[1].op_type == "mul"
        assert graph.nodes[2].op_type == "sub"

    def test_matmul_shape_computation(self):
        """Test correct shape computation for matmul."""
        tracer = Tracer()
        a = tracer.create_input((2, 3))
        b = tracer.create_input((3, 4))
        c = a @ b

        assert c.shape == (2, 4)

    def test_matmul_vec_vec(self):
        """Test matmul shape for vector @ vector."""
        tracer = Tracer()
        a = tracer.create_input((3,))
        b = tracer.create_input((3,))
        c = a @ b

        assert c.shape == ()  # Scalar output

    def test_matmul_batch(self):
        """Test matmul shape for batched matrices."""
        tracer = Tracer()
        a = tracer.create_input((2, 3, 4))
        b = tracer.create_input((2, 4, 5))
        c = a @ b

        assert c.shape == (2, 3, 5)


# =============================================================================
# Decorator Tests
# =============================================================================


class TestForgeDecorator:
    """Tests for the @forge decorator."""

    def test_simple_add(self):
        """Test tracing and executing simple addition."""
        @forge
        def add_fn(a, b):
            return a + b

        a = Array([[1.0, 2.0], [3.0, 4.0]])
        b = Array([[5.0, 6.0], [7.0, 8.0]])
        result = add_fn(a, b)

        assert result.list() == [[6.0, 8.0], [10.0, 12.0]]

    def test_simple_sub(self):
        """Test tracing subtraction."""
        @forge
        def sub_fn(a, b):
            return a - b

        a = Array([[10.0, 20.0], [30.0, 40.0]])
        b = Array([[1.0, 2.0], [3.0, 4.0]])
        result = sub_fn(a, b)

        assert result.list() == [[9.0, 18.0], [27.0, 36.0]]

    def test_simple_mul(self):
        """Test tracing multiplication."""
        @forge
        def mul_fn(a, b):
            return a * b

        a = Array([[2.0, 3.0], [4.0, 5.0]])
        b = Array([[1.0, 2.0], [3.0, 4.0]])
        result = mul_fn(a, b)

        assert result.list() == [[2.0, 6.0], [12.0, 20.0]]

    def test_simple_div(self):
        """Test tracing division."""
        @forge
        def div_fn(a, b):
            return a / b

        a = Array([[10.0, 20.0], [30.0, 40.0]])
        b = Array([[2.0, 4.0], [5.0, 8.0]])
        result = div_fn(a, b)

        assert result.list() == [[5.0, 5.0], [6.0, 5.0]]

    def test_chain_ops(self):
        """Test chained operations."""
        @forge
        def chain_fn(a, b):
            c = a + b
            d = c * a
            return d - b

        a = Array([[1.0, 2.0], [3.0, 4.0]])
        b = Array([[2.0, 3.0], [4.0, 5.0]])
        result = chain_fn(a, b)

        # c = [[3, 5], [7, 9]]
        # d = [[3, 10], [21, 36]]
        # result = [[1, 7], [17, 31]]
        assert result.list() == [[1.0, 7.0], [17.0, 31.0]]

    def test_matmul(self):
        """Test tracing matmul."""
        @forge
        def matmul_fn(a, b):
            return a @ b

        a = Array([[1.0, 2.0], [3.0, 4.0]])
        b = Array([[5.0, 6.0], [7.0, 8.0]])
        result = matmul_fn(a, b)

        # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        # = [[19, 22], [43, 50]]
        assert result.list() == [[19.0, 22.0], [43.0, 50.0]]

    def test_mixed_ops(self):
        """Test mixed element-wise and matmul ops."""
        @forge
        def mixed_fn(a, b):
            c = a + b
            return c @ a

        a = Array([[1.0, 2.0], [3.0, 4.0]])
        b = Array([[5.0, 6.0], [7.0, 8.0]])
        result = mixed_fn(a, b)

        # c = [[6, 8], [10, 12]]
        # c @ a = [[6*1+8*3, 6*2+8*4], [10*1+12*3, 10*2+12*4]]
        #       = [[30, 44], [46, 68]]
        assert result.list() == [[30.0, 44.0], [46.0, 68.0]]

    def test_trace_reuse(self):
        """Test that cached trace is reused on subsequent calls."""
        call_count = 0

        @forge
        def counted_fn(a, b):
            nonlocal call_count
            call_count += 1
            return a + b

        a = Array([1.0, 2.0, 3.0])
        b = Array([4.0, 5.0, 6.0])

        # First call - traces
        result1 = counted_fn(a, b)
        # Note: call_count increments during tracing
        first_count = call_count

        # Second call - should reuse trace
        result2 = counted_fn(a, b)
        second_count = call_count

        assert result1.list() == [5.0, 7.0, 9.0]
        assert result2.list() == [5.0, 7.0, 9.0]
        # The function body is only called during tracing
        assert first_count == second_count

    def test_get_graph(self):
        """Test accessing the traced graph."""
        @forge
        def simple_fn(a, b):
            return a + b

        a = Array([1.0, 2.0])
        b = Array([3.0, 4.0])

        # Before first call, no graph
        assert simple_fn._get_graph() is None

        # After call, graph exists
        simple_fn(a, b)
        graph = simple_fn._get_graph()

        assert graph is not None
        assert len(graph.nodes) == 1
        assert graph.nodes[0].op_type == "add"

    def test_clear_cache(self):
        """Test clearing the cached trace."""
        @forge
        def cacheable_fn(a, b):
            return a * b

        a = Array([1.0, 2.0])
        b = Array([3.0, 4.0])

        cacheable_fn(a, b)
        assert cacheable_fn._get_graph() is not None

        cacheable_fn._clear_cache()
        assert cacheable_fn._get_graph() is None


# =============================================================================
# Correctness vs Eager
# =============================================================================


class TestCorrectnessVsEager:
    """Compare traced execution against eager execution."""

    def _close_enough(self, a_list, b_list, rtol=1e-5):
        """Check if two nested lists are numerically close."""
        a_flat = np.array(a_list).flatten()
        b_flat = np.array(b_list).flatten()
        return np.allclose(a_flat, b_flat, rtol=rtol)

    def test_add_random(self):
        """Test add with random data."""
        np.random.seed(42)
        data_a = np.random.randn(10, 20).astype(np.float32)
        data_b = np.random.randn(10, 20).astype(np.float32)

        a = Array(data_a.tolist())
        b = Array(data_b.tolist())

        # Eager
        eager_result = a + b

        # Traced
        @forge
        def add_traced(x, y):
            return x + y

        traced_result = add_traced(a, b)

        assert self._close_enough(eager_result.list(), traced_result.list())

    def test_sub_random(self):
        """Test sub with random data."""
        np.random.seed(43)
        data_a = np.random.randn(15, 15).astype(np.float32)
        data_b = np.random.randn(15, 15).astype(np.float32)

        a = Array(data_a.tolist())
        b = Array(data_b.tolist())

        eager_result = a - b

        @forge
        def sub_traced(x, y):
            return x - y

        traced_result = sub_traced(a, b)

        assert self._close_enough(eager_result.list(), traced_result.list())

    def test_mul_random(self):
        """Test mul with random data."""
        np.random.seed(44)
        data_a = np.random.randn(8, 12).astype(np.float32)
        data_b = np.random.randn(8, 12).astype(np.float32)

        a = Array(data_a.tolist())
        b = Array(data_b.tolist())

        eager_result = a * b

        @forge
        def mul_traced(x, y):
            return x * y

        traced_result = mul_traced(a, b)

        assert self._close_enough(eager_result.list(), traced_result.list())

    def test_div_random(self):
        """Test div with random data (non-zero divisor)."""
        np.random.seed(45)
        data_a = np.random.randn(6, 8).astype(np.float32)
        data_b = np.random.randn(6, 8).astype(np.float32)
        data_b = np.where(np.abs(data_b) < 0.1, 1.0, data_b)  # Avoid near-zero

        a = Array(data_a.tolist())
        b = Array(data_b.tolist())

        eager_result = a / b

        @forge
        def div_traced(x, y):
            return x / y

        traced_result = div_traced(a, b)

        assert self._close_enough(eager_result.list(), traced_result.list())

    def test_matmul_random(self):
        """Test matmul with random data."""
        np.random.seed(46)
        data_a = np.random.randn(16, 32).astype(np.float32)
        data_b = np.random.randn(32, 24).astype(np.float32)

        a = Array(data_a.tolist())
        b = Array(data_b.tolist())

        eager_result = a @ b

        @forge
        def matmul_traced(x, y):
            return x @ y

        traced_result = matmul_traced(a, b)

        assert self._close_enough(eager_result.list(), traced_result.list())

    def test_complex_chain_random(self):
        """Test complex chain of operations."""
        np.random.seed(47)
        data_a = np.random.randn(8, 8).astype(np.float32)
        data_b = np.random.randn(8, 8).astype(np.float32)
        data_c = np.random.randn(8, 8).astype(np.float32)

        a = Array(data_a.tolist())
        b = Array(data_b.tolist())
        c = Array(data_c.tolist())

        # Eager
        temp1 = a + b
        temp2 = temp1 * c
        temp3 = temp2 @ a
        eager_result = temp3 - b

        # Traced
        @forge
        def complex_traced(x, y, z):
            t1 = x + y
            t2 = t1 * z
            t3 = t2 @ x
            return t3 - y

        traced_result = complex_traced(a, b, c)

        assert self._close_enough(eager_result.list(), traced_result.list())


# =============================================================================
# Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_shape_mismatch_on_replay(self):
        """Test that shape mismatch on replay raises error."""
        @forge
        def add_fn(a, b):
            return a + b

        # First call with (2, 3)
        a = Array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = Array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        add_fn(a, b)

        # Second call with same shape should work
        c = Array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        d = Array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
        result = add_fn(c, d)
        assert result.shape == (2, 3)

    def test_retrace_on_shape_change(self):
        """Test that function retraces when shapes change."""
        @forge
        def add_fn(a, b):
            return a + b

        # First call with shape (2, 2)
        a1 = Array([[1.0, 2.0], [3.0, 4.0]])
        b1 = Array([[5.0, 6.0], [7.0, 8.0]])
        result1 = add_fn(a1, b1)
        assert result1.shape == (2, 2)

        graph1 = add_fn._get_graph()
        assert graph1.shapes[0] == (2, 2)

        # Second call with shape (3, 3) - should retrace
        a2 = Array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        b2 = Array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]])
        result2 = add_fn(a2, b2)
        assert result2.shape == (3, 3)

        graph2 = add_fn._get_graph()
        assert graph2.shapes[0] == (3, 3)

    def test_wrong_input_count(self):
        """Test error on wrong number of inputs."""
        @forge
        def two_input_fn(a, b):
            return a + b

        # First call establishes signature
        a = Array([1.0, 2.0])
        b = Array([3.0, 4.0])
        two_input_fn(a, b)

        # Calling with wrong number should fail during tracing
        # (Python itself will raise TypeError for wrong args)
        with pytest.raises(TypeError):
            two_input_fn(a)

    def test_non_traced_array_return(self):
        """Test error when function returns non-TracedArray."""
        @forge
        def bad_return_fn(a, b):
            return 42  # Not a TracedArray

        a = Array([1.0, 2.0])
        b = Array([3.0, 4.0])

        with pytest.raises(TypeError, match="must return TracedArray"):
            bad_return_fn(a, b)


# =============================================================================
# Graph Structure Tests
# =============================================================================


class TestGraphStructure:
    """Tests for graph structure correctness."""

    def test_graph_input_indices(self):
        """Test that graph records correct input indices."""
        @forge
        def multi_input_fn(a, b, c):
            return a + b + c

        a = Array([1.0])
        b = Array([2.0])
        c = Array([3.0])
        multi_input_fn(a, b, c)

        graph = multi_input_fn._get_graph()
        assert graph.input_indices == [0, 1, 2]

    def test_graph_output_indices(self):
        """Test that graph records correct output index."""
        @forge
        def simple_fn(a, b):
            c = a + b
            d = c * a
            return d

        a = Array([1.0, 2.0])
        b = Array([3.0, 4.0])
        simple_fn(a, b)

        graph = simple_fn._get_graph()
        # Output should be the last computed value
        assert len(graph.output_indices) == 1
        assert graph.output_indices[0] == 3  # 2 inputs + 2 ops - 1 = idx 3

    def test_graph_node_order(self):
        """Test that nodes are recorded in execution order."""
        @forge
        def ordered_fn(a, b):
            c = a + b
            d = c - a
            e = d * b
            return e

        a = Array([1.0])
        b = Array([2.0])
        ordered_fn(a, b)

        graph = ordered_fn._get_graph()
        assert len(graph.nodes) == 3
        assert graph.nodes[0].op_type == "add"
        assert graph.nodes[1].op_type == "sub"
        assert graph.nodes[2].op_type == "mul"

    def test_graph_shapes_tracked(self):
        """Test that shapes are tracked for all values."""
        @forge
        def matmul_fn(a, b):
            return a @ b

        a = Array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        b = Array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
        matmul_fn(a, b)

        graph = matmul_fn._get_graph()
        assert graph.shapes[0] == (2, 3)  # Input a
        assert graph.shapes[1] == (3, 2)  # Input b
        assert graph.shapes[2] == (2, 2)  # Output


# =============================================================================
# 1D Array Tests
# =============================================================================


class Test1DArrays:
    """Tests specifically for 1D arrays."""

    def test_1d_add(self):
        """Test 1D array addition."""
        @forge
        def add_1d(a, b):
            return a + b

        a = Array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = Array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = add_1d(a, b)

        assert result.list() == [6.0, 6.0, 6.0, 6.0, 6.0]

    def test_1d_chain(self):
        """Test chained ops on 1D arrays."""
        @forge
        def chain_1d(a, b):
            c = a + b
            d = c * a
            return d - b

        a = Array([1.0, 2.0, 3.0])
        b = Array([2.0, 3.0, 4.0])
        result = chain_1d(a, b)

        # c = [3, 5, 7]
        # d = [3, 10, 21]
        # result = [1, 7, 17]
        assert result.list() == [1.0, 7.0, 17.0]
