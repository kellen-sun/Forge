#include <gtest/gtest.h>

#include "graph_io.h"
#include "../../cpp/include/memory_arena.h"

#ifndef FORGE_GTEST_DIR
#define FORGE_GTEST_DIR "../tests/gtest"
#endif

class MemoryArenaTest : public ::testing::TestWithParam<std::string> {};

TEST_P(MemoryArenaTest, TestAllocation) {
    std::string test_name = GetParam();
    std::string dir = std::string(FORGE_GTEST_DIR) + "/memory_arena_tests/";
    std::string in_path = dir + test_name + ".in";
    std::string out_path = dir + test_name + ".out";

    Graph g = parse_graph(in_path);
    auto lines = read_file_lines(out_path);

    MemoryArena m(g, 4);

    EXPECT_EQ(m.get_total_bytes(), std::stoull(lines[0]))
        << "Total bytes mismatch in " << test_name;
    EXPECT_EQ(m.get_all_offsets(), parse_line_to_vector<uint64_t>(lines[1]))
        << "Offsets mismatch in " << test_name;
    EXPECT_EQ(m.get_roots(), parse_line_to_vector<uint64_t>(lines[2]))
        << "Roots mismatch in " << test_name;
}

INSTANTIATE_TEST_SUITE_P(TestSuite, MemoryArenaTest,
                         ::testing::Values("test1", "test2", "test3", "test4"));
