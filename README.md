![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/kellen-sun/Forge.svg)
[![CI Build](https://github.com/kellen-sun/Forge/actions/workflows/main.yml/badge.svg)](https://github.com/kellen-sun/Forge/actions/workflows/main.yml)
<!---Add another badge here about version number? software license--->
# Forge
> Forge crafts Metal: an Array framework with eager execution and JIT graph compilation for Apple Silicon GPUs

Forge was initially intended as a Python library to run Metal Kernels on Apple Silicon (M-series) GPUs. Through it's development, it's picked up more features of general array/tensor libraries like ``numpy``, `pytorch`, and `mlx`.

### Then why build Forge?
I built Forge and continue to work on it primarily because it's a passion project with a lot to learn from. In writing the code for this project, I learned about software design, compilers, GPU kernels, Python libraries, and testing. I hope to learn much more as I continue working on the project. I also believe that in due time, certain features could be done better than in existing available frameworks.

### Features:
1. Eager Execution: operations run asynchronously, seamlessly on GPU with a familiar API design
2. JIT Graph Compilation (WIP): functions are compiled and optimized using the ``@forge`` decorator to be re-run more efficiently

### Example:
This simple code snippet below will add the two arrays using your GPU.
```py
from Forge import Array
a1 = Array([[1.0, 2.0], [3.0, 4.0]])
a2 = Array([[4.0, 5.0], [6.0, 7.0]])
result = a1 + a2
```

### Further information
If this sounds interesting and you'd like to try out the library, you can install it and run it on your own macbook using [this guide](https://github.com/kellen-sun/Forge/tree/main/docs/user/doc.md).

If you find anything interesting around and wish to contribute feel free to. You can take a look at [this guide](https://github.com/kellen-sun/Forge/tree/main/docs/developer/doc.md) to get setup and contribute or shoot me a message @kellen05 on discord.

Thanks be to our contributors!
- [@BillJJ](https://github.com/BillJJ)
- [@DawDa07](https://github.com/DawDa07)
