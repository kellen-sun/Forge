#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../../cpp/include/memory_arena.h"

template <typename T>
std::vector<T> parse_line_to_vector(const std::string& line) {
    std::istringstream iss(line);
    std::vector<T> vec;
    T val;
    while (iss >> val) vec.push_back(val);
    return vec;
}

std::vector<std::string> read_file_lines(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open test file: " + filepath);
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        lines.push_back(line);
    }
    return lines;
}

class MemoryArenaTest : public ::testing::TestWithParam<std::string> {
   protected:
    Graph parse_graph(const std::string& filepath) {
        auto lines = read_file_lines(filepath);
        int output_index = std::stoi(lines.back());
        lines.pop_back();

        std::vector<Node> nodes;
        for (size_t i = 0; i < lines.size(); i += 6) {
            Node n;
            n.op = static_cast<OpCode>(std::stoi(lines[i]));
            n.inputs = parse_line_to_vector<int>(lines[i + 1]);
            n.shape = parse_line_to_vector<int64_t>(lines[i + 2]);
            n.strides = parse_line_to_vector<int64_t>(lines[i + 3]);
            n.offset = std::stoll(lines[i + 4]);
            n.args = parse_line_to_vector<int64_t>(lines[i + 5]);
            nodes.push_back(n);
        }
        return Graph(nodes, output_index);
    }
};

TEST_P(MemoryArenaTest, TestAllocation) {
    std::string test_name = GetParam();

    std::string in_path = "../tests/gtest/memory_arena_tests/" + test_name + ".in";
    std::string out_path = "../tests/gtest/memory_arena_tests/" + test_name + ".out";

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

INSTANTIATE_TEST_SUITE_P(TestSuite, MemoryArenaTest, ::testing::Values("test1", "test2"));
