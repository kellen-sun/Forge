# A transpiler and runtime for a Pythonic DSL that compiles to Metal Shading Language (MSL) and executes GPU kernels from within Python.
This project aims to be a subset of what numba is to CUDA for Apple Silicon. It aims to transpile numpy like code (and more in the future) directly to Metal at run-time and execute it on the GPU. It'll start as being similar to mlx and diversify beyond (hopefully). This will allow for decorating numpy like functions with @metal, to directly run them on GPU.

It currently also includes ./denoiser folder, which is a simple code snippet aimed at removing background noise. Currently, aiming to be able to transpile that.

DSL Language Description:


Personal To-Do:
1. Add more features to the transpiler, constants, binops, etc.
2. Learn about what comes after the transpilation, how to manage resources to GPU, when to read/write from it and CPU
3. Check-up on ideas and methods from other similar projects like mlx
