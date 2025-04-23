# A transpiler and runtime for a Pythonic DSL that compiles to Metal Shading Language (MSL) and executes GPU kernels from within Python.
This project aims to be a subset of what numba is to CUDA for Apple Silicon. It aims to transpile numpy like code (and more in the future) directly to Metal at run-time and execute it on the GPU. It'll start as being similar to mlx and diversify beyond (hopefully). This will allow for decorating numpy like functions with @metal, to directly run them on GPU.

It currently also includes ./denoiser folder, which is a simple code snippet aimed at removing background noise. Currently, aiming to be able to transpile that.

DSL Language Description:
1. Currently must take in numpy float32 arrays of the same length as input and return that as output.
2. Also supports binary operations, +, -, *, /
3. Supports constants (interpreted as an array full with that constant)
4. Supports sum(), len()

Personal To-Do:
1. Add more features to the transpiler: matrix mults
2. Learn about what comes after the transpilation, how to manage resources to GPU, when to read/write from it and CPU
3. Check-up on ideas and methods from other similar projects like mlx
4. deducing types: so that sum(a)*sum(b) gets treated as scalar mult not vector (traverse AST, to label types?)
