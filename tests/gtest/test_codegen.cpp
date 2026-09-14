#include <gtest/gtest.h>

#include <cstdlib>

#include "../../cpp/include/compiler.h"
#include "graph_io.h"
#include "ir_builders.h"

#ifndef FORGE_GTEST_DIR
#define FORGE_GTEST_DIR "../tests/gtest"
#endif
#ifndef FORGE_SOURCE_DIR
#define FORGE_SOURCE_DIR ".."
#endif

static std::vector<std::string> quoted_strings_in_list(const std::string& text, const std::string& marker,
                                                       char open, char close) {
    const auto mark = text.find(marker);
    if (mark == std::string::npos) return {};
    const auto begin = text.find(open, mark);
    if (begin == std::string::npos) return {};
    const auto end = text.find(close, begin + 1);
    if (end == std::string::npos) return {};
    const std::string region = text.substr(begin, end - begin);
    std::vector<std::string> names;
    for (size_t i = 0; i < region.size(); ++i) {
        if (region[i] != '"') continue;
        const auto j = region.find('"', i + 1);
        if (j == std::string::npos) break;
        names.push_back(region.substr(i + 1, j - i - 1));
        i = j;
    }
    return names;
}

class CodegenGoldenTest : public ::testing::TestWithParam<std::string> {};

TEST_P(CodegenGoldenTest, MatchesGolden) {
    std::string test_name = GetParam();
    std::string dir = std::string(FORGE_GTEST_DIR) + "/codegen_tests/";
    std::string in_path = dir + test_name + ".in";
    std::string out_path = dir + test_name + ".out";

    Graph g = parse_graph(in_path);
    IR ir;
    ir.nodes = std::move(g.nodes);
    ir.output_index = g.output_index;
    optimize_graph(ir);
    g.nodes = std::move(ir.nodes);
    g.output_index = ir.output_index;
    generateKernels(g);
    std::string actual = dump_codegen(g);

    if (std::getenv("UPDATE_GOLDENS")) {
        std::ofstream out(out_path, std::ios::trunc);
        if (!out) throw std::runtime_error("Failed to write golden: " + out_path);
        out << actual;
        return;
    }

    EXPECT_EQ(actual, read_file(out_path))
        << "codegen golden mismatch in " << test_name
        << " (rerun with UPDATE_GOLDENS=1 to rewrite)";
}

INSTANTIATE_TEST_SUITE_P(TestSuite, CodegenGoldenTest,
                         ::testing::Values("identity", "add_2x2", "add_const", "view_add",
                                           "dce_dead_add", "canonicalize_identity_reshape",
                                           "sum_global", "sum_axis", "unary_exp"));

TEST(Codegen, UnaryOpsPythonMatchesCommonH) {
    const std::string py = read_file(std::string(FORGE_SOURCE_DIR) + "/py/Forge/ops.py");
    const std::string cc = read_file(std::string(FORGE_SOURCE_DIR) + "/cpp/include/common.h");
    const auto from_py = quoted_strings_in_list(py, "UNARY_OPS", '[', ']');
    const auto from_h = quoted_strings_in_list(cc, "kUnaryNames", '{', '}');
    ASSERT_FALSE(from_py.empty());
    ASSERT_EQ(from_py, from_h) << "keep py/Forge/ops.py UNARY_OPS in lockstep with "
                                  "kUnaryNames in cpp/include/common.h";
    ASSERT_EQ(static_cast<int>(from_h.size()), kUnaryCount);
    for (int i = 0; i < kUnaryCount; ++i) {
        EXPECT_STREQ(from_h[static_cast<size_t>(i)].c_str(), kUnaryNames[i]);
    }
}

TEST(Codegen, UnaryKindsEmitNamedKernels) {
    for (int kind = 0; kind < kUnaryCount; ++kind) {
        Node u;
        u.op = OpCode::UNARY;
        u.inputs = {0};
        u.shape = {2};
        u.strides = {1};
        u.offset = 0;
        u.args = {static_cast<int64_t>(kind)};

        Graph g{{make_input({2}, {1}), u}, 1};
        generateKernels(g);
        const std::string want = std::string("Out[gid]=") + kUnaryNames[kind] + "(A[idx_a]);";
        EXPECT_NE(g.shader_source.find(want), std::string::npos) << kUnaryNames[kind];
        EXPECT_EQ(g.configs[1].name, std::string("op_1_") + kUnaryNames[kind]);
    }
}
