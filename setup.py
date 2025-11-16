from setuptools import setup, find_packages

setup(
    name="Forge",
    version="0.0.1",
    packages=find_packages(where="py/Forge"),
    package_dir={"": "py"},
)
