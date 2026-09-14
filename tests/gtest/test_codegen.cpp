#include <gtest/gtest.h>

#include <cstdlib>

#include "../../cpp/include/compiler.h"
#include "graph_io.h"
#include "ir_builders.h"

#ifndef FORGE_GTEST_DIR
#define FORGE_GTEST_DIR "../tests/gtest"
#endif

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

TEST(Codegen, ZerosEmitsOutputOnlyFillKernel) {
    Node zeros;
    zeros.op = OpCode::ZEROS;
    zeros.shape = {2, 3};
    zeros.strides = {3, 1};
    zeros.offset = 0;

    Graph g{{zeros}, 0};
    generateKernels(g);

    ASSERT_EQ(g.configs.size(), 1u);
    EXPECT_EQ(g.configs[0].name, "op_0_zeros");
    EXPECT_NE(g.shader_source.find("device float* Out [[buffer(0)]]"), std::string::npos);
    EXPECT_NE(g.shader_source.find("Out[gid]=0.0f;"), std::string::npos);
}
