# Documentation to use Forge
Read the README.md first for general information.

## Installation Process -> Currently same as for Dev

Clone the repo, install dependencies and build the library. In the future, this will be just a single pip install command.
```
git clone repo
```

Make a virtual environment and activate it. 
```
pip install pytest
```
To run benchmarks probably will need to pip install other stuff like pytorch, numba, mlx, etc. (tbd)

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

setup the Forge library itself (from project root run):
```
pip install -e .
```

to run tests
```
pytest -q
```
