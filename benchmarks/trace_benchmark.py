"""
Benchmark for tracing-based JIT vs eager execution.

Compares:
1. Simple ops (varying sizes)
2. Chained ops (4 operations)
3. Long chain (10 operations)
4. Matmul chain
5. Tracing overhead (first call vs subsequent)
"""

import time
from typing import Callable, Tuple

import numpy as np
from Forge import Array, forge


def time_fn(fn: Callable, warmup: int = 3, iterations: int = 20) -> Tuple[float, float]:
    """Time a function with warmup iterations. Returns (mean_ms, std_ms)."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_table_header():
    print(
        f"  {'Test':<30} | {'Eager':>12} | {'Traced':>12} | {'Speedup':>10}"
    )
    print("  " + "-" * 75)


def print_row(name: str, eager_t: float, traced_t: float):
    speedup = eager_t / traced_t if traced_t > 0 else float('inf')
    print(
        f"  {name:<30} | {eager_t:>10.3f}ms | {traced_t:>10.3f}ms | {speedup:>9.2f}x"
    )


def benchmark_simple_ops():
    """Benchmark simple addition at various sizes."""
    print_header("SIMPLE ADDITION (varying sizes)")
    print_table_header()

    sizes = [64, 256, 512, 1024, 2048]

    for n in sizes:
        data_a = np.random.rand(n, n).astype(np.float32)
        data_b = np.random.rand(n, n).astype(np.float32)

        a = Array(data_a.tolist())
        b = Array(data_b.tolist())

        # Eager
        def eager_add():
            return a + b

        eager_time, _ = time_fn(eager_add)

        # Traced (with warmup to ensure traced)
        @forge
        def traced_add(x, y):
            return x + y

        # Prime the trace
        traced_add(a, b)

        def run_traced():
            return traced_add(a, b)

        traced_time, _ = time_fn(run_traced)

        print_row(f"{n}x{n} add", eager_time, traced_time)


def benchmark_chain_4_ops():
    """Benchmark a chain of 4 operations."""
    print_header("CHAIN OF 4 OPERATIONS (a + b - c * d)")
    print_table_header()

    sizes = [256, 512, 1024, 2048]

    for n in sizes:
        data_a = np.random.rand(n, n).astype(np.float32)
        data_b = np.random.rand(n, n).astype(np.float32)
        data_c = np.random.rand(n, n).astype(np.float32)
        data_d = np.random.rand(n, n).astype(np.float32)

        a = Array(data_a.tolist())
        b = Array(data_b.tolist())
        c = Array(data_c.tolist())
        d = Array(data_d.tolist())

        # Eager
        def eager_chain():
            t1 = a + b
            t2 = t1 - c
            return t2 * d

        eager_time, _ = time_fn(eager_chain)

        # Traced
        @forge
        def traced_chain(x, y, z, w):
            t1 = x + y
            t2 = t1 - z
            return t2 * w

        traced_chain(a, b, c, d)

        def run_traced():
            return traced_chain(a, b, c, d)

        traced_time, _ = time_fn(run_traced)

        print_row(f"{n}x{n} chain-4", eager_time, traced_time)


def benchmark_long_chain():
    """Benchmark a long chain of 10 operations."""
    print_header("LONG CHAIN (10 OPERATIONS)")
    print_table_header()

    sizes = [256, 512, 1024]

    for n in sizes:
        data_a = np.random.rand(n, n).astype(np.float32)
        data_b = np.random.rand(n, n).astype(np.float32)

        a = Array(data_a.tolist())
        b = Array(data_b.tolist())

        # Eager
        def eager_long():
            t = a + b
            t = t - a
            t = t * b
            t = t + a
            t = t - b
            t = t * a
            t = t + b
            t = t - a
            t = t * b
            return t + a

        eager_time, _ = time_fn(eager_long)

        # Traced
        @forge
        def traced_long(x, y):
            t = x + y
            t = t - x
            t = t * y
            t = t + x
            t = t - y
            t = t * x
            t = t + y
            t = t - x
            t = t * y
            return t + x

        traced_long(a, b)

        def run_traced():
            return traced_long(a, b)

        traced_time, _ = time_fn(run_traced)

        print_row(f"{n}x{n} chain-10", eager_time, traced_time)


def benchmark_matmul_chain():
    """Benchmark matmul-heavy computation."""
    print_header("MATMUL CHAIN ((a + b) @ c)")
    print_table_header()

    sizes = [128, 256, 512, 1024]

    for n in sizes:
        data_a = np.random.rand(n, n).astype(np.float32)
        data_b = np.random.rand(n, n).astype(np.float32)
        data_c = np.random.rand(n, n).astype(np.float32)

        a = Array(data_a.tolist())
        b = Array(data_b.tolist())
        c = Array(data_c.tolist())

        # Eager
        def eager_matmul():
            t = a + b
            return t @ c

        eager_time, _ = time_fn(eager_matmul)

        # Traced
        @forge
        def traced_matmul(x, y, z):
            t = x + y
            return t @ z

        traced_matmul(a, b, c)

        def run_traced():
            return traced_matmul(a, b, c)

        traced_time, _ = time_fn(run_traced)

        print_row(f"{n}x{n} matmul-chain", eager_time, traced_time)


def benchmark_tracing_overhead():
    """Measure overhead of tracing vs subsequent calls."""
    print_header("TRACING OVERHEAD (first call vs subsequent)")
    print(f"  {'Size':<20} | {'First (trace)':>15} | {'Subsequent':>15} | {'Overhead':>10}")
    print("  " + "-" * 70)

    sizes = [256, 512, 1024]

    for n in sizes:
        data_a = np.random.rand(n, n).astype(np.float32)
        data_b = np.random.rand(n, n).astype(np.float32)

        a = Array(data_a.tolist())
        b = Array(data_b.tolist())

        @forge
        def trace_fn(x, y):
            t = x + y
            t = t * x
            t = t - y
            return t + x

        # Measure first call (includes tracing)
        first_times = []
        for _ in range(5):
            trace_fn._clear_cache()
            start = time.perf_counter()
            trace_fn(a, b)
            end = time.perf_counter()
            first_times.append((end - start) * 1000)
        first_time = np.mean(first_times)

        # Measure subsequent calls (cached trace)
        subsequent_times = []
        for _ in range(20):
            start = time.perf_counter()
            trace_fn(a, b)
            end = time.perf_counter()
            subsequent_times.append((end - start) * 1000)
        subsequent_time = np.mean(subsequent_times)

        overhead = first_time - subsequent_time
        print(
            f"  {n}x{n:<17} | {first_time:>13.3f}ms | {subsequent_time:>13.3f}ms | {overhead:>8.3f}ms"
        )


def benchmark_multiple_outputs():
    """Benchmark functions with multiple outputs (when supported)."""
    print_header("ELEMENT-WISE OPERATIONS BREAKDOWN")
    print_table_header()

    n = 1024
    data_a = np.random.rand(n, n).astype(np.float32)
    data_b = np.random.rand(n, n).astype(np.float32)

    a = Array(data_a.tolist())
    b = Array(data_b.tolist())

    ops = [
        ("add", lambda: a + b),
        ("sub", lambda: a - b),
        ("mul", lambda: a * b),
        ("div", lambda: a / b),
        ("matmul", lambda: a @ b),
    ]

    for op_name, eager_fn in ops:
        eager_time, _ = time_fn(eager_fn)

        if op_name != "matmul":
            @forge
            def traced_op(x, y, _op=op_name):
                if _op == "add":
                    return x + y
                elif _op == "sub":
                    return x - y
                elif _op == "mul":
                    return x * y
                elif _op == "div":
                    return x / y

        else:
            @forge
            def traced_op(x, y):
                return x @ y

        traced_op(a, b)

        def run_traced():
            return traced_op(a, b)

        traced_time, _ = time_fn(run_traced)

        print_row(f"{n}x{n} {op_name}", eager_time, traced_time)


def run_all():
    print("\n" + "=" * 80)
    print(" TRACING-BASED JIT BENCHMARK: Eager vs Traced Execution")
    print(" All times in milliseconds (lower is better)")
    print(" Speedup > 1.0 means traced is faster")
    print("=" * 80)

    benchmark_simple_ops()
    benchmark_chain_4_ops()
    benchmark_long_chain()
    benchmark_matmul_chain()
    benchmark_tracing_overhead()
    benchmark_multiple_outputs()

    print("\n" + "=" * 80)
    print(" BENCHMARK COMPLETE")
    print("=" * 80)
    print("\nNotes:")
    print("- MVP implementation: traced execution uses same backend ops as eager")
    print("- Expected: similar performance (Python dispatch overhead present in both)")
    print("- Future versions will batch ops to backend for reduced Python overhead")
    print()


if __name__ == "__main__":
    run_all()
