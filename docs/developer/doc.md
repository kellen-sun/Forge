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

### To run benchmarks
```
pip install numpy torch mlx
```
Then simply run the Python files in ``/benchmarks``

### To run the code formatter
Every git commit will run the pre-commit code formatter. You can also run it manually with:
```
pre-commit run --all-files
```
