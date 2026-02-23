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

include(GoogleTest)
gtest_discover_tests(forge_tests)
