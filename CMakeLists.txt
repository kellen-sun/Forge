cmake_minimum_required(VERSION 3.15)
project(forge_backend LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(
    _backend 
    cpp/src/bindings.cpp
    cpp/src/array_handle.cpp
    cpp/src/compiler.cpp
    cpp/src/runtime.cpp
)

# Put output in python package
set_target_properties(_backend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/py/Forge
)
