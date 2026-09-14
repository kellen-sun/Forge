# Documentation to use Forge
Read the README.md first for general information.

## Installation Process
Run ``pip install forge-metal``.

## The Library:
The main provided type is ``Array`` which is a tensor type wrapping a GPU side buffer. It can be created from an array('f'), memoryview or numpy (``Array.from_buffer(mv, shape)``) from Python and a shape or directly from nested lists/tuples (``Array([...])``).

In Python, we can save those ``Array`` types and apply operations on them such as ``a + b`` which is a pointwise addition. We can also ask for the underlying list or shape back ``a.shape`` and ``a.list()``.

We can index into the Array with all the usual methods, with the brackets [4] supporting both regular indexing and slicing [1:5:2] and into multiple dimensions just as in usual lists [3, 4]. When indexing to read the items, this merely creates a view into the already existing data (without making a copy). -> Later on, we can support fancy indexing with double brackets [[4, 5]].

We also support ``len()`` and ``sum()/.sum()``. We can take a transpose using ``Array.T`` and reshape our array with ``Array.reshape()``, using a ``-1`` to fill in a dimension. Note that transposes never make a copy of the underlying data, while reshape usually doesn't, but might if the data to be reshaped is not contiguous in memory.

## ``@forge`` (WIP)
Decorating a function with ``@forge`` traces it once for a given input shape/strides/offset, generates Metal kernels, and reuses that graph on later calls.

```py
from Forge import Array, forge

@forge
def f(a, b):
    return (a + b) * 2.0

print(f(Array([1.0, 2.0]), Array([3.0, 4.0])).list())
```

Currently this covers:

- elementwise add/sub/mul/div (including scalar constants and reverse ops like `2.0 * x`)
- elementwise unaries: `exp`, `exp2`, `exp10`, `log`, `log2`, `log10`, `sqrt`, `rsqrt`, `abs`, `sign`, `ceil`, `floor`, `round`, `trunc`, `fract`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh` (methods on the array or `Forge.exp(x)`, etc.)
- reductions: `.sum()` / `.sum(axis=..., keepdims=...)`
- views, reshape, and transpose
- `zeros`
- random factories: `rand` and `randn` (with runtime seed progression)
- indexed assignment inside `@forge`: `x[key] = value` for scalar or matching-shape, non-overlapping RHS
- arithmetic assignment inside `@forge`: `+=`, `-=`, `*=`, `/=`

Not yet compiled: matmul (`@`)

