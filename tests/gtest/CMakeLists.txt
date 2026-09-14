include(FetchContent)

# pull latest stable GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(forge_tests
    test_memory_arena.cpp
    test_array_helpers.cpp
    test_constant_args.cpp
    test_codegen.cpp
    test_py_cpp_abi.cpp
    test_verify.cpp
    test_dce.cpp
    test_canonicalize.cpp
    test_shape_infer.cpp
)

set_target_properties(forge_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
)

# link your main C++ library and gtest
target_link_libraries(forge_tests
    PRIVATE
    forge_lib
    GTest::gtest_main
)

target_compile_definitions(forge_tests PRIVATE
    FORGE_GTEST_DIR="${CMAKE_SOURCE_DIR}/tests/gtest"
    FORGE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

include(GoogleTest)
gtest_discover_tests(forge_tests)
