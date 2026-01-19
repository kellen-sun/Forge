"""
Benchmark comparing @forge traced execution vs eager execution.

Timing methodology:
- Each timed function includes GPU sync (via element access which calls item() -> synchronize())
- This matches the pattern in compare_frameworks.py: (result)[0, 0] forces GPU completion
- Warmup runs before timing to exclude JIT compilation overhead from measurements
"""

import time

import numpy as np
from Forge import Array, forge


def time_fn(fn, warmup: int = 3, iterations: int = 10) -> tuple[float, float]:
    """
    Time a function with warmup iterations. Returns (mean_ms, std_ms).

    The function should include GPU sync internally (e.g., by accessing result[0,0]).
    """
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


def verify_close(eager_result, traced_result, name: str):
    """Verify eager and traced results match."""
    eager_list = eager_result.list()
    traced_list = traced_result.list()
    eager_np = np.array(eager_list, dtype=np.float32)
    traced_np = np.array(traced_list, dtype=np.float32)
    if not np.allclose(eager_np, traced_np, rtol=1e-4, atol=1e-5):
        print(f"  WARNING: {name} results don't match!")
        return False
    return True


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_table_header():
    print(f"  {'Operation':<35} | {'Eager':>10} | {'Traced':>10} | {'Ratio':>8}")
    print("  " + "-" * 73)


def print_row(name: str, eager_ms: float, traced_ms: float):
    ratio = traced_ms / eager_ms if eager_ms > 0 else float("inf")
    # ratio < 1 means traced is faster, > 1 means eager is faster
    indicator = "faster" if ratio < 0.95 else ("slower" if ratio > 1.05 else "~same")
    print(f"  {name:<35} | {eager_ms:>8.3f}ms | {traced_ms:>8.3f}ms | {ratio:>5.2f}x {indicator}")


# ============================================================================
# Benchmark: Simple Add
# ============================================================================

def eager_add(a, b):
    return a + b


@forge
def traced_add(a, b):
    return a + b


def benchmark_add():
    """Benchmark simple addition."""
    print_header("SIMPLE ADDITION: a + b")
    print_table_header()

    sizes = [
        ((1000,), "[0]"),
        ((10000,), "[0]"),
        ((100000,), "[0]"),
        ((100, 100), "[0,0]"),
        ((500, 500), "[0,0]"),
        ((1000, 1000), "[0,0]"),
    ]

    for shape, idx_str in sizes:
        data_a = np.random.rand(*shape).astype(np.float32)
        data_b = np.random.rand(*shape).astype(np.float32)
        a = Array(data_a.tolist())
        b = Array(data_b.tolist())

        # Verify correctness first
        eager_r = eager_add(a, b)
        traced_r = traced_add(a, b)
        verify_close(eager_r, traced_r, f"add {shape}")

        # Time with sync: access element to force GPU completion
        # Match exact pattern from compare_frameworks.py
        if len(shape) == 1:
            eager_time, _ = time_fn(lambda: eager_add(a, b)[0])
            traced_time, _ = time_fn(lambda: traced_add(a, b)[0])
        else:
            eager_time, _ = time_fn(lambda: eager_add(a, b)[0, 0])
            traced_time, _ = time_fn(lambda: traced_add(a, b)[0, 0])

        print_row(f"add {shape}", eager_time, traced_time)


# ============================================================================
# Benchmark: 4-Operation Chain
# ============================================================================

def eager_chain4(a, b):
    c = a + b
    d = c * a
    e = d - b
    return e / a


@forge
def traced_chain4(a, b):
    c = a + b
    d = c * a
    e = d - b
    return e / a


def benchmark_chain4():
    """Benchmark 4-operation chain: ((a+b)*a - b) / a"""
    print_header("4-OP CHAIN: ((a+b)*a - b) / a")
    print_table_header()

    sizes = [
        ((10000,), "[0]"),
        ((100000,), "[0]"),
        ((500, 500), "[0,0]"),
        ((1000, 1000), "[0,0]"),
    ]

    for shape, idx_str in sizes:
        # Avoid division by zero
        data_a = (np.random.rand(*shape) + 0.1).astype(np.float32)
        data_b = (np.random.rand(*shape) + 0.1).astype(np.float32)
        a = Array(data_a.tolist())
        b = Array(data_b.tolist())

        # Verify correctness
        eager_r = eager_chain4(a, b)
        traced_r = traced_chain4(a, b)
        verify_close(eager_r, traced_r, f"chain4 {shape}")

        # Time with sync
        if len(shape) == 1:
            eager_time, _ = time_fn(lambda: eager_chain4(a, b)[0])
            traced_time, _ = time_fn(lambda: traced_chain4(a, b)[0])
        else:
            eager_time, _ = time_fn(lambda: eager_chain4(a, b)[0, 0])
            traced_time, _ = time_fn(lambda: traced_chain4(a, b)[0, 0])

        print_row(f"chain4 {shape}", eager_time, traced_time)


# ============================================================================
# Benchmark: 8-Operation Chain
# ============================================================================

def eager_chain8(a, b):
    x = a + b
    x = x * a
    x = x - b
    x = x / a
    x = x + b
    x = x * a
    x = x - b
    return x / a


@forge
def traced_chain8(a, b):
    x = a + b
    x = x * a
    x = x - b
    x = x / a
    x = x + b
    x = x * a
    x = x - b
    return x / a


def benchmark_chain8():
    """Benchmark 8-operation chain."""
    print_header("8-OP CHAIN: repeated (+, *, -, /)")
    print_table_header()

    sizes = [
        ((10000,), "[0]"),
        ((100000,), "[0]"),
        ((500, 500), "[0,0]"),
    ]

    for shape, idx_str in sizes:
        data_a = (np.random.rand(*shape) + 0.1).astype(np.float32)
        data_b = (np.random.rand(*shape) + 0.1).astype(np.float32)
        a = Array(data_a.tolist())
        b = Array(data_b.tolist())

        # Verify correctness
        eager_r = eager_chain8(a, b)
        traced_r = traced_chain8(a, b)
        verify_close(eager_r, traced_r, f"chain8 {shape}")

        # Time with sync
        if len(shape) == 1:
            eager_time, _ = time_fn(lambda: eager_chain8(a, b)[0])
            traced_time, _ = time_fn(lambda: traced_chain8(a, b)[0])
        else:
            eager_time, _ = time_fn(lambda: eager_chain8(a, b)[0, 0])
            traced_time, _ = time_fn(lambda: traced_chain8(a, b)[0, 0])

        print_row(f"chain8 {shape}", eager_time, traced_time)


# ============================================================================
# Benchmark: Matrix Multiplication
# ============================================================================

def eager_matmul(a, b):
    return a @ b


@forge
def traced_matmul(a, b):
    return a @ b


def benchmark_matmul():
    """Benchmark matrix multiplication."""
    print_header("MATRIX MULTIPLICATION: a @ b")
    print_table_header()

    sizes = [128, 256, 512, 1024]

    for n in sizes:
        data_a = np.random.rand(n, n).astype(np.float32)
        data_b = np.random.rand(n, n).astype(np.float32)
        a = Array(data_a.tolist())
        b = Array(data_b.tolist())

        # Verify correctness
        eager_r = eager_matmul(a, b)
        traced_r = traced_matmul(a, b)
        verify_close(eager_r, traced_r, f"matmul {n}x{n}")

        # Time with sync - exact pattern from compare_frameworks.py
        eager_time, _ = time_fn(lambda: (a @ b)[0, 0])
        traced_time, _ = time_fn(lambda: traced_matmul(a, b)[0, 0])

        print_row(f"matmul {n}x{n}", eager_time, traced_time)


# ============================================================================
# Benchmark: Matmul Chain (3 matmuls)
# ============================================================================

def eager_matmul_chain(a, b):
    c = a @ b
    d = c @ a
    return d @ b


@forge
def traced_matmul_chain(a, b):
    c = a @ b
    d = c @ a
    return d @ b


def benchmark_matmul_chain():
    """Benchmark chain of 3 matrix multiplications."""
    print_header("MATMUL CHAIN: (a @ b) @ a @ b")
    print_table_header()

    sizes = [128, 256, 512]

    for n in sizes:
        data_a = np.random.rand(n, n).astype(np.float32)
        data_b = np.random.rand(n, n).astype(np.float32)
        a = Array(data_a.tolist())
        b = Array(data_b.tolist())

        # Verify correctness
        eager_r = eager_matmul_chain(a, b)
        traced_r = traced_matmul_chain(a, b)
        verify_close(eager_r, traced_r, f"matmul_chain {n}x{n}")

        # Time with sync
        eager_time, _ = time_fn(lambda: eager_matmul_chain(a, b)[0, 0])
        traced_time, _ = time_fn(lambda: traced_matmul_chain(a, b)[0, 0])

        print_row(f"3x matmul {n}x{n}", eager_time, traced_time)


# ============================================================================
# Benchmark: Mixed Operations (matmul + elementwise)
# ============================================================================

def eager_mixed(a, b):
    c = a @ b
    d = c + a
    e = d * b
    return e @ a


@forge
def traced_mixed(a, b):
    c = a @ b
    d = c + a
    e = d * b
    return e @ a


def benchmark_mixed():
    """Benchmark mixed matmul and elementwise operations."""
    print_header("MIXED OPS: ((a @ b) + a) * b @ a")
    print_table_header()

    sizes = [128, 256, 512]

    for n in sizes:
        data_a = np.random.rand(n, n).astype(np.float32)
        data_b = np.random.rand(n, n).astype(np.float32)
        a = Array(data_a.tolist())
        b = Array(data_b.tolist())

        # Verify correctness
        eager_r = eager_mixed(a, b)
        traced_r = traced_mixed(a, b)
        verify_close(eager_r, traced_r, f"mixed {n}x{n}")

        # Time with sync
        eager_time, _ = time_fn(lambda: eager_mixed(a, b)[0, 0])
        traced_time, _ = time_fn(lambda: traced_mixed(a, b)[0, 0])

        print_row(f"mixed {n}x{n}", eager_time, traced_time)


# ============================================================================
# Benchmark: Compilation Overhead
# ============================================================================

def benchmark_compilation():
    """Measure compilation overhead: first call vs cached calls."""
    print_header("COMPILATION OVERHEAD (first call includes tracing + graph creation)")
    print(f"  {'Shape':<25} | {'1st call':>12} | {'Cached':>12} | {'Overhead':>12}")
    print("  " + "-" * 68)

    sizes = [(1000,), (10000,), (256, 256), (512, 512)]

    for shape in sizes:
        data_a = np.random.rand(*shape).astype(np.float32)
        data_b = np.random.rand(*shape).astype(np.float32)
        a = Array(data_a.tolist())
        b = Array(data_b.tolist())

        # Create fresh function to measure compilation
        @forge
        def fresh_fn(x, y):
            z = x + y
            z = z * x
            return z - y

        # First call: includes tracing + backend graph creation
        start = time.perf_counter()
        if len(shape) == 1:
            _ = fresh_fn(a, b)[0]
        else:
            _ = fresh_fn(a, b)[0, 0]
        first_ms = (time.perf_counter() - start) * 1000

        # Subsequent calls: cached
        if len(shape) == 1:
            cached_ms, _ = time_fn(lambda: fresh_fn(a, b)[0], warmup=0)
        else:
            cached_ms, _ = time_fn(lambda: fresh_fn(a, b)[0, 0], warmup=0)

        overhead_ms = first_ms - cached_ms
        print(f"  {str(shape):<25} | {first_ms:>10.3f}ms | {cached_ms:>10.3f}ms | {overhead_ms:>10.3f}ms")


# ============================================================================
# Main
# ============================================================================

def run_all():
    print("\n" + "=" * 80)
    print(" TRACED vs EAGER BENCHMARK")
    print(" ")
    print(" Timing: Each operation includes GPU sync (element access forces completion)")
    print(" Ratio: traced_time / eager_time (<1 = traced faster, >1 = eager faster)")
    print("=" * 80)

    benchmark_add()
    benchmark_chain4()
    benchmark_chain8()
    benchmark_matmul()
    benchmark_matmul_chain()
    benchmark_mixed()
    benchmark_compilation()

    print("\n" + "=" * 80)
    print(" SUMMARY")
    print("=" * 80)
    print("""
 MVP Performance Notes:
 - Traced execution replays operations via Python dispatch (same as eager)
 - Expected: traced ~= eager (no fusion optimization yet)
 - Small overhead possible from graph lookup + Python loop in execute()

 Future optimizations (will show speedups):
 - v0.3: Batch ops to C++ backend (reduce Python overhead)
 - v0.4: Kernel fusion (fuse elementwise chains into single GPU kernel)
""")


if __name__ == "__main__":
    run_all()
