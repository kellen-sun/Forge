"""
Comprehensive matmul benchmarks for Forge.

This module tests:
1. Correctness - Verifies matmul produces correct results matching NumPy
2. Performance - Compares Forge (Metal) vs NumPy across various sizes
3. Edge cases - Strided arrays, transposed inputs, batched operations
"""

import time
from typing import Callable

import numpy as np
from Forge import Array

# ==============================================================================
# Utility functions
# ==============================================================================


def time_fn(fn: Callable, warmup: int = 2, iterations: int = 10) -> tuple[float, float]:
    """Time a function with warmup iterations.

    Returns:
        tuple of (mean_time, std_time) in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        fn()

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return np.mean(times), np.std(times)


def allclose(forge_result: Array, numpy_result: np.ndarray, rtol: float = 1e-4) -> bool:
    """Check if Forge and NumPy results match within tolerance."""
    forge_np = np.array(forge_result.list(), dtype=np.float32)
    return np.allclose(forge_np, numpy_result, rtol=rtol, atol=1e-5)


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_result(name: str, forge_time: float, numpy_time: float, correct: bool):
    """Print benchmark result in a formatted way."""
    speedup = numpy_time / forge_time if forge_time > 0 else float("inf")
    status = "PASS" if correct else "FAIL"
    print(
        f"  {name:<40} | Forge: {forge_time:8.3f}ms | NumPy: {numpy_time:8.3f}ms | "
        f"Speedup: {speedup:6.2f}x | {status}"
    )


# ==============================================================================
# Correctness Tests
# ==============================================================================


def test_correctness():
    """Run correctness tests for various matmul scenarios."""
    print_header("CORRECTNESS TESTS")

    all_passed = True

    # Test 1: Basic 2D matmul
    print("\n  [2D Matrix Multiplication]")
    for m, k, n in [(2, 3, 2), (10, 10, 10), (64, 64, 64), (128, 256, 128)]:
        a_np = np.random.rand(m, k).astype(np.float32)
        b_np = np.random.rand(k, n).astype(np.float32)
        a_forge = Array(a_np.tolist())
        b_forge = Array(b_np.tolist())

        result_forge = a_forge @ b_forge
        result_np = a_np @ b_np

        passed = allclose(result_forge, result_np)
        all_passed &= passed
        status = "PASS" if passed else "FAIL"
        print(f"    ({m}x{k}) @ ({k}x{n}) -> ({m}x{n}): {status}")

    # Test 2: Matrix-vector multiplication
    print("\n  [Matrix-Vector Multiplication]")
    for m, k in [(10, 5), (64, 32), (256, 128)]:
        a_np = np.random.rand(m, k).astype(np.float32)
        b_np = np.random.rand(k).astype(np.float32)
        a_forge = Array(a_np.tolist())
        b_forge = Array(b_np.tolist())

        result_forge = a_forge @ b_forge
        result_np = a_np @ b_np

        passed = allclose(result_forge, result_np)
        all_passed &= passed
        status = "PASS" if passed else "FAIL"
        print(f"    ({m}x{k}) @ ({k},) -> ({m},): {status}")

    # Test 3: Vector-matrix multiplication
    print("\n  [Vector-Matrix Multiplication]")
    for k, n in [(5, 10), (32, 64), (128, 256)]:
        a_np = np.random.rand(k).astype(np.float32)
        b_np = np.random.rand(k, n).astype(np.float32)
        a_forge = Array(a_np.tolist())
        b_forge = Array(b_np.tolist())

        result_forge = a_forge @ b_forge
        result_np = a_np @ b_np

        passed = allclose(result_forge, result_np)
        all_passed &= passed
        status = "PASS" if passed else "FAIL"
        print(f"    ({k},) @ ({k}x{n}) -> ({n},): {status}")

    # Test 4: Vector-vector (dot product)
    print("\n  [Vector Dot Product]")
    for k in [10, 100, 1000]:
        a_np = np.random.rand(k).astype(np.float32)
        b_np = np.random.rand(k).astype(np.float32)
        a_forge = Array(a_np.tolist())
        b_forge = Array(b_np.tolist())

        result_forge = a_forge @ b_forge
        result_np = a_np @ b_np

        # Scalar comparison
        forge_val = result_forge.list()
        passed = abs(forge_val - result_np) < 1e-3
        all_passed &= passed
        status = "PASS" if passed else "FAIL"
        print(f"    ({k},) @ ({k},) -> scalar: {status}")

    # Test 5: Batched matmul
    print("\n  [Batched Matrix Multiplication]")
    for batch, m, k, n in [(2, 4, 4, 4), (4, 16, 16, 16), (8, 32, 32, 32)]:
        a_np = np.random.rand(batch, m, k).astype(np.float32)
        b_np = np.random.rand(batch, k, n).astype(np.float32)
        a_forge = Array(a_np.tolist())
        b_forge = Array(b_np.tolist())

        result_forge = a_forge @ b_forge
        result_np = a_np @ b_np

        passed = allclose(result_forge, result_np)
        all_passed &= passed
        status = "PASS" if passed else "FAIL"
        print(
            f"    ({batch}x{m}x{k}) @ ({batch}x{k}x{n}) -> ({batch}x{m}x{n}): {status}"
        )

    # Test 6: Transposed input
    print("\n  [Transposed Input]")
    for m, k, n in [(10, 20, 15), (64, 128, 64)]:
        a_np = np.random.rand(k, m).astype(np.float32)  # Will transpose to (m, k)
        b_np = np.random.rand(k, n).astype(np.float32)
        a_forge = Array(a_np.tolist())
        b_forge = Array(b_np.tolist())

        result_forge = a_forge.T @ b_forge
        result_np = a_np.T @ b_np

        passed = allclose(result_forge, result_np)
        all_passed &= passed
        status = "PASS" if passed else "FAIL"
        print(f"    ({k}x{m}).T @ ({k}x{n}) -> ({m}x{n}): {status}")

    # Test 7: Strided arrays
    print("\n  [Strided Arrays]")
    a_np = np.random.rand(8, 8).astype(np.float32)
    b_np = np.random.rand(4, 4).astype(np.float32)
    a_forge = Array(a_np.tolist())
    b_forge = Array(b_np.tolist())

    # Row stride
    result_forge = a_forge[::2, :4] @ b_forge
    result_np = a_np[::2, :4] @ b_np
    passed = allclose(result_forge, result_np)
    all_passed &= passed
    status = "PASS" if passed else "FAIL"
    print(f"    Row strided [::2, :4] @ full: {status}")

    # Column stride
    result_forge = a_forge[:4, ::2] @ b_forge
    result_np = a_np[:4, ::2] @ b_np
    passed = allclose(result_forge, result_np)
    all_passed &= passed
    status = "PASS" if passed else "FAIL"
    print(f"    Col strided [:4, ::2] @ full: {status}")

    print(f"\n  Overall correctness: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


# ==============================================================================
# Performance Benchmarks
# ==============================================================================


def benchmark_square_matrices():
    """Benchmark square matrix multiplication at various sizes."""
    print_header("SQUARE MATRIX BENCHMARK (NxN @ NxN)")
    print(f"  {'Size':<40} | {'Forge':>12} | {'NumPy':>12} | {'Speedup':>10} | Status")
    print("  " + "-" * 90)

    sizes = [32, 64, 128, 256, 512, 1024, 2048]

    for n in sizes:
        a_np = np.random.rand(n, n).astype(np.float32)
        b_np = np.random.rand(n, n).astype(np.float32)
        a_forge = Array(a_np.tolist())
        b_forge = Array(b_np.tolist())

        # Time Forge
        forge_time, _ = time_fn(lambda: a_forge @ b_forge, warmup=2, iterations=5)

        # Time NumPy
        numpy_time, _ = time_fn(lambda: a_np @ b_np, warmup=2, iterations=5)

        # Verify correctness
        result_forge = a_forge @ b_forge
        result_np = a_np @ b_np
        correct = allclose(result_forge, result_np)

        print_result(f"{n}x{n} @ {n}x{n}", forge_time, numpy_time, correct)


def benchmark_rectangular_matrices():
    """Benchmark rectangular matrix multiplication."""
    print_header("RECTANGULAR MATRIX BENCHMARK")
    print(f"  {'Size':<40} | {'Forge':>12} | {'NumPy':>12} | {'Speedup':>10} | Status")
    print("  " + "-" * 90)

    shapes = [
        ((1024, 256), (256, 1024)),  # Wide @ Tall
        ((256, 1024), (1024, 256)),  # Tall @ Wide
        ((2048, 512), (512, 128)),  # Large @ Medium
        ((128, 2048), (2048, 128)),  # Medium inner dim
        ((4096, 128), (128, 4096)),  # Very wide/tall
    ]

    for (m, k), (_, n) in shapes:
        a_np = np.random.rand(m, k).astype(np.float32)
        b_np = np.random.rand(k, n).astype(np.float32)
        a_forge = Array(a_np.tolist())
        b_forge = Array(b_np.tolist())

        forge_time, _ = time_fn(lambda: a_forge @ b_forge, warmup=2, iterations=5)
        numpy_time, _ = time_fn(lambda: a_np @ b_np, warmup=2, iterations=5)

        result_forge = a_forge @ b_forge
        result_np = a_np @ b_np
        correct = allclose(result_forge, result_np)

        print_result(f"({m}x{k}) @ ({k}x{n})", forge_time, numpy_time, correct)


def benchmark_batched_matmul():
    """Benchmark batched matrix multiplication."""
    print_header("BATCHED MATRIX BENCHMARK")
    print(f"  {'Size':<40} | {'Forge':>12} | {'NumPy':>12} | {'Speedup':>10} | Status")
    print("  " + "-" * 90)

    configs = [
        (4, 256, 256),
        (8, 256, 256),
        (16, 128, 128),
        (32, 64, 64),
        (4, 512, 512),
    ]

    for batch, m, n in configs:
        a_np = np.random.rand(batch, m, n).astype(np.float32)
        b_np = np.random.rand(batch, n, m).astype(np.float32)
        a_forge = Array(a_np.tolist())
        b_forge = Array(b_np.tolist())

        forge_time, _ = time_fn(lambda: a_forge @ b_forge, warmup=2, iterations=5)
        numpy_time, _ = time_fn(lambda: a_np @ b_np, warmup=2, iterations=5)

        result_forge = a_forge @ b_forge
        result_np = a_np @ b_np
        correct = allclose(result_forge, result_np)

        print_result(
            f"batch={batch}, {m}x{n} @ {n}x{m}", forge_time, numpy_time, correct
        )


def benchmark_matvec():
    """Benchmark matrix-vector multiplication."""
    print_header("MATRIX-VECTOR BENCHMARK")
    print(f"  {'Size':<40} | {'Forge':>12} | {'NumPy':>12} | {'Speedup':>10} | Status")
    print("  " + "-" * 90)

    sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]

    for m, k in sizes:
        a_np = np.random.rand(m, k).astype(np.float32)
        b_np = np.random.rand(k).astype(np.float32)
        a_forge = Array(a_np.tolist())
        b_forge = Array(b_np.tolist())

        forge_time, _ = time_fn(lambda: a_forge @ b_forge, warmup=2, iterations=5)
        numpy_time, _ = time_fn(lambda: a_np @ b_np, warmup=2, iterations=5)

        result_forge = a_forge @ b_forge
        result_np = a_np @ b_np
        correct = allclose(result_forge, result_np)

        print_result(f"({m}x{k}) @ ({k},)", forge_time, numpy_time, correct)


def benchmark_transposed():
    """Benchmark transposed matrix operations."""
    print_header("TRANSPOSED INPUT BENCHMARK")
    print(f"  {'Size':<40} | {'Forge':>12} | {'NumPy':>12} | {'Speedup':>10} | Status")
    print("  " + "-" * 90)

    sizes = [256, 512, 1024, 2048]

    for n in sizes:
        # A.T @ B
        a_np = np.random.rand(n, n).astype(np.float32)
        b_np = np.random.rand(n, n).astype(np.float32)
        a_forge = Array(a_np.tolist())
        b_forge = Array(b_np.tolist())

        forge_time, _ = time_fn(lambda: a_forge.T @ b_forge, warmup=2, iterations=5)
        numpy_time, _ = time_fn(lambda: a_np.T @ b_np, warmup=2, iterations=5)

        result_forge = a_forge.T @ b_forge
        result_np = a_np.T @ b_np
        correct = allclose(result_forge, result_np)

        print_result(f"({n}x{n}).T @ ({n}x{n})", forge_time, numpy_time, correct)


# ==============================================================================
# Main entry point
# ==============================================================================


def run_all_benchmarks():
    """Run all benchmark suites."""
    print("\n" + "=" * 70)
    print(" FORGE MATMUL BENCHMARK SUITE")
    print(" Testing correctness and performance of Metal-accelerated matmul")
    print("=" * 70)

    # Correctness first
    correctness_passed = test_correctness()

    if not correctness_passed:
        print("\n  WARNING: Some correctness tests failed!")
        print("  Performance benchmarks may not be meaningful.\n")

    # Performance benchmarks
    benchmark_square_matrices()
    benchmark_rectangular_matrices()
    benchmark_batched_matmul()
    benchmark_matvec()
    benchmark_transposed()

    print("\n" + "=" * 70)
    print(" BENCHMARK COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_benchmarks()
