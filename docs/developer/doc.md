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
pip install pre-commit
```

tools for the backend
```
brew install pybind11
brew install cmake
```

to build the backend
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

to run tracer tests only
```
pytest tests/test_tracer.py -v
```

to run benchmarks
```
pip install numpy torch mlx
```

to run trace benchmark (eager vs traced execution)
```
python benchmarks/trace_benchmark.py
```

Every git commit will run the pre-commit code formatter. You can also run it manually with:
```
pre-commit run --all-files
```
