"""
Comprehensive matmul benchmark comparing Forge, NumPy, PyTorch (MPS), and MLX.
"""

import time

import mlx.core as mx
import numpy as np
import torch

from Forge import Array


def time_fn(fn, warmup: int = 3, iterations: int = 10) -> tuple[float, float]:
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
    print("\n" + "=" * 90)
    print(f" {title}")
    print("=" * 90)


def print_table_header():
    print(
        f"  {'Size':<25} | {'Forge':>10} | {'NumPy':>10} | {'PyTorch':>10} | {'MLX':>10} | Best"
    )
    print("  " + "-" * 85)


def print_row(name: str, forge_t: float, numpy_t: float, torch_t: float, mlx_t: float):
    times = {"Forge": forge_t, "NumPy": numpy_t, "PyTorch": torch_t, "MLX": mlx_t}
    best = min(times, key=times.get)
    print(
        f"  {name:<25} | {forge_t:>8.3f}ms | {numpy_t:>8.3f}ms | "
        f"{torch_t:>8.3f}ms | {mlx_t:>8.3f}ms | {best}"
    )


def benchmark_square_matrices():
    """Benchmark square matrix multiplication."""
    print_header("SQUARE MATRIX BENCHMARK (NxN @ NxN)")
    print_table_header()

    sizes = [128, 256, 512, 1024, 2048, 4096]

    for n in sizes:
        # Create data
        data_a = np.random.rand(n, n).astype(np.float32)
        data_b = np.random.rand(n, n).astype(np.float32)

        # Forge
        forge_a = Array(data_a.tolist())
        forge_b = Array(data_b.tolist())
        forge_time, _ = time_fn(lambda: forge_a @ forge_b)

        # NumPy
        numpy_time, _ = time_fn(lambda: data_a @ data_b)

        # PyTorch MPS
        torch_a = torch.from_numpy(data_a).to("mps")
        torch_b = torch.from_numpy(data_b).to("mps")

        def torch_matmul():
            result = torch_a @ torch_b
            torch.mps.synchronize()
            return result

        torch_time, _ = time_fn(torch_matmul)

        # MLX
        mlx_a = mx.array(data_a)
        mlx_b = mx.array(data_b)

        def mlx_matmul():
            result = mlx_a @ mlx_b
            mx.eval(result)
            return result

        mlx_time, _ = time_fn(mlx_matmul)

        print_row(f"{n}x{n}", forge_time, numpy_time, torch_time, mlx_time)


def benchmark_rectangular_matrices():
    """Benchmark rectangular matrix multiplication."""
    print_header("RECTANGULAR MATRIX BENCHMARK")
    print_table_header()

    shapes = [
        ((1024, 256), (256, 1024)),
        ((256, 1024), (1024, 256)),
        ((2048, 512), (512, 256)),
        ((4096, 256), (256, 4096)),
    ]

    for (m, k), (_, n) in shapes:
        data_a = np.random.rand(m, k).astype(np.float32)
        data_b = np.random.rand(k, n).astype(np.float32)

        # Forge
        forge_a = Array(data_a.tolist())
        forge_b = Array(data_b.tolist())
        forge_time, _ = time_fn(lambda: forge_a @ forge_b)

        # NumPy
        numpy_time, _ = time_fn(lambda: data_a @ data_b)

        # PyTorch MPS
        torch_a = torch.from_numpy(data_a).to("mps")
        torch_b = torch.from_numpy(data_b).to("mps")

        def torch_matmul():
            result = torch_a @ torch_b
            torch.mps.synchronize()
            return result

        torch_time, _ = time_fn(torch_matmul)

        # MLX
        mlx_a = mx.array(data_a)
        mlx_b = mx.array(data_b)

        def mlx_matmul():
            result = mlx_a @ mlx_b
            mx.eval(result)
            return result

        mlx_time, _ = time_fn(mlx_matmul)

        print_row(f"({m}x{k})@({k}x{n})", forge_time, numpy_time, torch_time, mlx_time)


def benchmark_batched():
    """Benchmark batched matrix multiplication."""
    print_header("BATCHED MATRIX BENCHMARK")
    print_table_header()

    configs = [
        (4, 512, 512),
        (8, 256, 256),
        (16, 256, 256),
        (32, 128, 128),
    ]

    for batch, m, n in configs:
        data_a = np.random.rand(batch, m, n).astype(np.float32)
        data_b = np.random.rand(batch, n, m).astype(np.float32)

        # Forge
        forge_a = Array(data_a.tolist())
        forge_b = Array(data_b.tolist())
        forge_time, _ = time_fn(lambda: forge_a @ forge_b)

        # NumPy
        numpy_time, _ = time_fn(lambda: data_a @ data_b)

        # PyTorch MPS
        torch_a = torch.from_numpy(data_a).to("mps")
        torch_b = torch.from_numpy(data_b).to("mps")

        def torch_matmul():
            result = torch_a @ torch_b
            torch.mps.synchronize()
            return result

        torch_time, _ = time_fn(torch_matmul)

        # MLX
        mlx_a = mx.array(data_a)
        mlx_b = mx.array(data_b)

        def mlx_matmul():
            result = mlx_a @ mlx_b
            mx.eval(result)
            return result

        mlx_time, _ = time_fn(mlx_matmul)

        print_row(f"batch={batch}, {m}x{n}", forge_time, numpy_time, torch_time, mlx_time)


def benchmark_transformer_shapes():
    """Benchmark shapes common in transformer models."""
    print_header("TRANSFORMER-LIKE SHAPES")
    print_table_header()

    # Common transformer dimensions
    # batch, seq_len, d_model, d_ff
    configs = [
        ("Attention QK^T", (8, 512, 64), (8, 64, 512)),  # batch, heads, seq, head_dim
        ("Attention scores@V", (8, 512, 512), (8, 512, 64)),
        ("FFN up proj", (1, 512, 768), (1, 768, 3072)),
        ("FFN down proj", (1, 512, 3072), (1, 3072, 768)),
    ]

    for name, shape_a, shape_b in configs:
        data_a = np.random.rand(*shape_a).astype(np.float32)
        data_b = np.random.rand(*shape_b).astype(np.float32)

        # Forge
        forge_a = Array(data_a.tolist())
        forge_b = Array(data_b.tolist())
        forge_time, _ = time_fn(lambda: forge_a @ forge_b)

        # NumPy
        numpy_time, _ = time_fn(lambda: data_a @ data_b)

        # PyTorch MPS
        torch_a = torch.from_numpy(data_a).to("mps")
        torch_b = torch.from_numpy(data_b).to("mps")

        def torch_matmul():
            result = torch_a @ torch_b
            torch.mps.synchronize()
            return result

        torch_time, _ = time_fn(torch_matmul)

        # MLX
        mlx_a = mx.array(data_a)
        mlx_b = mx.array(data_b)

        def mlx_matmul():
            result = mlx_a @ mlx_b
            mx.eval(result)
            return result

        mlx_time, _ = time_fn(mlx_matmul)

        print_row(name, forge_time, numpy_time, torch_time, mlx_time)


def run_all():
    print("\n" + "=" * 90)
    print(" MATMUL FRAMEWORK COMPARISON: Forge vs NumPy vs PyTorch (MPS) vs MLX")
    print(" All times in milliseconds (lower is better)")
    print("=" * 90)

    benchmark_square_matrices()
    benchmark_rectangular_matrices()
    benchmark_batched()
    benchmark_transformer_shapes()

    print("\n" + "=" * 90)
    print(" BENCHMARK COMPLETE")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_all()
