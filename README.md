# A transpiler and runtime for a Pythonic DSL that compiles to Metal Shading Language (MSL) and executes GPU kernels from within Python.
This project aims to be a subset of what numba is to CUDA for Apple Silicon by transpiling numpy like code (and more in the future) directly to Metal at run-time and execute it on the GPU. It'll start as being similar to mlx and diversify beyond (hopefully). This will allow for decorating numpy like functions with @metal, to directly run them on GPU. 

Current performance: 20% ish improvement on matrix multiplication (example in main.py) including overhead. Over 10x faster, if not including overhead.

It currently also includes ./denoiser folder, which is a simple code snippet aimed at removing background noise. Currently, aiming to be able to transpile that.

DSL Language Description:
1. Currently must take in numpy float32 arrays of the same length as input and return that as output.
2. Also supports binary operations, +, -, *, /
3. Supports constants (interpreted as an array full with that constant)
4. Supports sum(), len()
5. Supports matmult (A @ B) when sizes match

Setup + How to Run this:
1. clone the repo
2. setup an appropriate venv and pip install the required libraries, namely metal
3. edit your main.py file to have the numpy ops you want
4. decorate the function with @metal
5. run it!

Personal To-Do:
1. Add more features to the transpiler: 
2. Learn about what comes after the transpilation, how to manage resources to GPU, when to read/write from it and CPU
3. Check-up on ideas and methods from other similar projects like mlx
4. Deal with unnecessary moves between buffers (input to temp then to out, then out to out) -> (matmult on inputs directly into out)
5. Expand supported types from float to double, int, etc.?
6. Avoid recreating pipelines. they can be made once at the start or something, but then should be cached (how? pickle?) for the same function (say matmult) pipeline should be the same