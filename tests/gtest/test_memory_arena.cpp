#include <gtest/gtest.h>

#include "../../cpp/include/memory_arena.h"

TEST(MemoryArenaTest, EmptyGraph) {
    std::vector<uint64_t> empty;
    Graph g({}, 0);
    MemoryArena m(g);

    EXPECT_EQ(m.get_total_bytes(), 0);
    EXPECT_EQ(m.get_all_offsets(), empty);
    EXPECT_EQ(m.get_roots(), empty);
}

TEST(MemoryArenaTest, SimpleGraph) {
    // def fn(a: 2x4, b: 4x2): return a @ b; both contiguous
    Node n1{OpCode::INPUT, {}, {2, 4}, {4, 1}, 0, {}};
    Node n2{OpCode::INPUT, {}, {4, 2}, {2, 1}, 0, {}};
    Node n3{OpCode::MATMUL, {0, 1}, {2, 2}, {2, 1}, 0, {}};
    Graph g({n1, n2, n3}, 2);
    MemoryArena m(g, 4);

    std::vector<uint64_t> expected_offsets{0, 0, 0};
    std::vector<uint64_t> expected_roots{0, 1, 2};

    EXPECT_EQ(m.get_total_bytes(), 0);
    EXPECT_EQ(m.get_all_offsets(), expected_offsets);
    EXPECT_EQ(m.get_roots(), expected_roots);
}

// bruh too much typing. lets just make a test fixture, takes in a list of files
// each one has a description of the graph. lines of numbers. 6 lines per node
// describes each input of the node
// parse through construct graph. then have an expected file as well
// can generate test cases by running tracer and printing the frontend "graph"
