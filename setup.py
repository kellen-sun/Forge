from setuptools import find_packages, setup

setup(
    name="Forge",
    version="0.0.1",
    packages=find_packages(where="py/Forge"),
    package_dir={"": "py"},
    install_requires=[
        "numpy>=1.20.0",
    ],
)
