# Documentation to setup and help write Forge
Read the README.md first for general information.

## Installation Process
Clone the repo:
```
git clone https://github.com/kellen-sun/Forge.git
```

Make a virtual environment and activate it:
```
python3 -m venv .venv
source .venv/bin/activate
pip install pytest
pip install pre-commit
```

Install tools for the backend:
```
pip install nanobind
brew install cmake
```
### Building the backend
Keep your venv activated, so that cmake can find ``nanobind``
```
mkdir build
cd build
```
#### Configure the build.

For pytest, and to use the Python library:
```
cmake ..
```
For ``gtest`` with -O0, Asan:
```
cmake .. -DBUILD_TYPE=Debug -DBUILD_TESTS=ON
```
For ``gtest`` with -O3:
```
cmake .. -DBUILD_TESTS=ON
```
#### Build.
```
cmake --build .
```

### To run tests
#### Pytest
Setup the Forge library itself (from project root run):
```
pip install -e .
```
To run all ``pytest``'s. The simplest command is:
```
pytest
```
To run ``pytest`` with validation of Metal Buffer uses, add this environment variable (only checks if buffer accesses are correct; does not check for leaks). ``-s`` to catch the Metal API error description:
```
MTL_DEBUG_LAYER=1 pytest -s
```
Refer to ``pytest`` documentation, to learn the commands to run specific files or tests at a time, and options etc.

#### Gtest
Compile with one of the ``gtest`` options mentioned above.
Run:
```
./build/tests/forge_tests
```
Refer to ``gtest`` documentation, to learn the commands to run specific files or tests at a time, and other options etc.

Python and C++ tables that must stay synced (opcodes, unary/nullary names, eager Metal kernel names) are checked by gtests in ``tests/gtest/test_py_cpp_abi.cpp``. GitHub CI builds with ``-DBUILD_TESTS=ON`` and runs ``./build/tests/forge_tests``. It does not run pytest.


##### Codegen goldens
``generateKernels`` is checked by file tests under ``tests/gtest/codegen_tests/`` (same graph ``.in`` format as the memory-arena tests). The ``.out`` files store dispatch configs plus compact MSL (one kernel per line). Tests compare that compact text byte-for-byte.

Rewrite goldens after an intentional emitter change (from the repo root, with gtests built):
```
UPDATE_GOLDENS=1 ./build/tests/forge_tests --gtest_filter='*CodegenGoldenTest*'
```

##### Viewing compact goldens
``tests/gtest/codegen_tests/view_golden.py`` pretty-prints a ``.out`` for humans. It does not change files or affect CI.

From the repo root:
```
# every *.out in codegen_tests/
python3 tests/gtest/codegen_tests/view_golden.py

# one or more specific goldens
python3 tests/gtest/codegen_tests/view_golden.py tests/gtest/codegen_tests/add_2x2.out
python3 tests/gtest/codegen_tests/view_golden.py tests/gtest/codegen_tests/add_const.out tests/gtest/codegen_tests/view_add.out
```

Configs (kernel name, grid, group) print as-is; the shader after ``---`` is indented so Metal ``[[buffer(...)]]`` attributes stay on the same line as the declaration.

### Compiler passes
``optimize_graph`` in ``cpp/src/compiler.cpp`` runs this order:

1. ``ShapeInferPass``
2. ``FunctionalizePass`` — each write gets a new version of its storage, and later reads (including through views) use that version
3. ``CanonicalizePass``
4. ``ConstantFoldPass``
5. ``CSEPass``
6. ``ElementwiseFusionPass`` — one walk; a pure elementwise chain becomes one kernel
7. ``DCEPass``
8. ``SnapshotStaleReadsPass`` — if a fused kernel would read a value an in-place write already replaced, copy the old buffer. Graphs with no in-place updates skip this

Codegen goldens run the same pipeline, then ``generateKernels``. A new pass has to be inserted in that list. Fusion does not run again after later rewrites, and the snapshot copy is the current way to keep fused reads correct across writes.

### To run benchmarks
```
pip install numpy torch mlx
```
Then simply run the Python files in ``/benchmarks``

### To run the code formatter
Install the hook once, from the repo root with the venv active:
```
pre-commit install
```
After that, every git commit runs the formatter. You can also run it manually with:
```
pre-commit run --all-files
```

### Cutting a PyPI release
The version is written in two places, and they have to match:

- ``pyproject.toml`` — ``version`` under ``[project]``. This is the version pip and the PyPI badge use.
- ``py/Forge/__init__.py`` — ``__version__``.

GitHub CI does not build or upload a package. From the repo root, on the commit you want to publish (a Mac, so the Metal extension can compile):
```
python3 -m pip install build twine
python3 -m build
```
That writes ``dist/forge_metal-<version>-….whl`` and ``dist/forge_metal-<version>.tar.gz``. The wheel tag matches the Python and macOS you built on. Other Python versions and older macOS install the source tarball and compile it.

Upload after that commit is on ``main``:
```
twine upload dist/*
```
PyPI will not replace a version that is already there, so each upload needs a new version. The package name stays ``forge-metal``.
