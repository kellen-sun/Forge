![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/kellen-sun/Forge.svg)
# Forge
> Forge crafts Metal: an Array framework with eager execution and JIT graph compilation for Apple Silicon GPUs

Forge was initially intended as a Python library to run Metal Kernels on Apple Silicon GPUs. Through it's development, it's picked up more features of general array/tensor libraries like ``numpy``, `pytorch`, and `mlx`.

Then why build Forge?

The interface of the library is mainly split into two parts, a section allowing asynchronous eager operations and one allowing for compiled functions through a ``@forge`` decorator (a WIP). For example, this code snippet below will add the two arrays using your GPU.
```py
from Forge import Array
a1 = Array([[1.0, 2.0], [3.0, 4.0]])
a2 = Array([[4.0, 5.0], [6.0, 7.0]])
result = a1 + a2
```

If this sounds interesting and you'd like to use the library, you can install it and run it on your own macbook using [this guide](https://github.com/kellen-sun/Forge/tree/main/docs/user/doc.md).

If you find anything interesting around and wish to contribute feel free to. You can take a look at [this guide](https://github.com/kellen-sun/Forge/tree/main/docs/developer/doc.md) to get setup and contribute or shoot me a message @kellen05 on discord.

Thanks be to our contributors!
- @BillJJ
- @DawDa07
