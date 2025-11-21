cmake_minimum_required(VERSION 3.20)
project(forge_backend LANGUAGES C CXX OBJC OBJCXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(
    _backend 
    cpp/src/bindings.cpp
    cpp/src/array_handle.mm
    cpp/src/array_add.mm
    cpp/src/forge_handle.mm
)

# Put output in python package
set_target_properties(_backend PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/py/Forge
)

target_link_libraries(_backend PRIVATE
    "-framework Metal"
    "-framework Foundation"
)
target_compile_options(_backend PRIVATE
    "-fobjc-arc"
)
