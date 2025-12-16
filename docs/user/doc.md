# Documentation to use Forge
Read the README.md first for general information.

## Installation Process -> Currently same as for developers

Clone the repo, install dependencies and build the library. In the future, this will be just a single pip install command.
```
git clone repo
```

Make a virtual environment and activate it.
```
brew install pybind11
brew install cmake
```

```
mkdir build
cd build
cmake ..
cmake --build .
```

Setup the Forge library itself (from project root run):
```
pip install -e .
```

## The Library:
The main provided type is ``Array`` which is a tensor type wrapping a GPU side buffer. It can be created from an array('f'), memoryview or numpy (``Array.from_buffer(mv, shape)``) from Python and a shape or directly from nested lists/tuples (``Array([...])``).

In Python, we can save those ``Array`` types and apply operations on them such as ``a + b`` which is a pointwise addition. We can also ask for the underlying list or shape back ``a.shape`` and ``a.list()``.

We can index into the Array with all the usual methods, with the brackets [4] supporting both regular indexing and slicing [1:5:2] and into multiple dimensions just as in usual lists [3, 4]. When indexing to read the items, this merely creates a view into the already existing data (without making a copy). -> Later on, we can support fancy indexing with double brackets [[4, 5]].

We also support ``len()`` and ``sum()``. We can take a transpose using ``Array.T`` and reshape our array with ``Array.reshape()``, using a ``-1`` to fill in columns.
