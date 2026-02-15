# Documentation to get setup and help write Forge
Read the README.md first for general information.

## Installation Process -> Currently same as for users

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
pip install nanobind
brew install cmake
```

to build the backend (with venv activated, to be able to find the nanobind cmake)
```
mkdir build
cd build
cmake ..
cmake -DBUILD_TYPE=Debug .
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

sometimes we need to validate Metal Buffer uses, so add this environment variable (only checks if buffer accesses are correct, in range, etc. not leaks). -s to catch the Metal API error description.
```
MTL_DEBUG_LAYER=1 pytest -s
```

to run benchmarks
```
pip install numpy torch mlx
```

Every git commit will run the pre-commit code formatter. You can also run it manually with:
```
pre-commit run --all-files
```
