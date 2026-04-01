cmake_minimum_required(VERSION 3.20)
project(forge_backend LANGUAGES C CXX OBJC OBJCXX)
if(NOT BUILD_TYPE)
    set(BUILD_TYPE Release CACHE STRING "Choose the type of build.")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

add_library(forge_lib STATIC
    cpp/src/array_binops.mm
    cpp/src/array_handle.mm
    cpp/src/array_unaryops.mm
    cpp/src/compiler.cpp
    cpp/src/compiler.mm
    cpp/src/forge_handle.mm
    cpp/src/memory_arena.cpp
    cpp/src/metal_utils.mm
    cpp/src/runtime.mm
)

target_link_libraries(forge_lib PUBLIC
    "-framework Metal"
    "-framework Foundation"
    "-framework MetalPerformanceShaders"
)
target_compile_options(forge_lib PUBLIC
    "-fobjc-arc"
)

if(BUILD_TYPE MATCHES "Debug")
    message(STATUS "Building in DEBUG mode")
    target_compile_options(forge_lib PUBLIC "-fsanitize=address" "-g" "-O0")
    target_link_options(forge_lib PUBLIC "-fsanitize=address")
else()
    message(STATUS "Building in RELEASE mode")
    nanobind_add_module(_backend
        cpp/src/bindings_impl.cpp
        cpp/src/bindings.cpp
    )
    target_compile_options(_backend PRIVATE "-O3")
    # Put output _backend in python package
    target_link_libraries(_backend PRIVATE forge_lib)
    set_target_properties(_backend PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/py/Forge
    )
endif()

# Build gtests
option(BUILD_TESTS "Build Forge C++ unit tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/gtest)
endif()
