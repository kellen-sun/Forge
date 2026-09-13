#include <gtest/gtest.h>

#include "../../cpp/include/compiler.h"
#include "../../cpp/include/ir.h"
#include "../../cpp/include/pass.h"
#include "ir_builders.h"

TEST(Canonicalize, RewritesIdentityLayoutThenDCEDropsIt) {
    IR ir;
    ir.nodes = {
        make_input({2, 2}, {2, 1}),
        make_layout(OpCode::RESHAPE, 0, {2, 2}, {2, 1}),
        make_add(1, 1, {2, 2}, {2, 1}),
    };
    ir.output_index = 2;

    CanonicalizePass{}.run(ir);
    EXPECT_EQ(ir.nodes[2].inputs, (std::vector<int>{0, 0}));
    EXPECT_EQ(ir.nodes[1].op, OpCode::RESHAPE);

    DCEPass{}.run(ir);
    ASSERT_EQ(ir.nodes.size(), 2u);
    EXPECT_EQ(ir.nodes[0].op, OpCode::INPUT);
    EXPECT_EQ(ir.nodes[1].op, OpCode::ADD);
    EXPECT_EQ(ir.nodes[1].inputs, (std::vector<int>{0, 0}));
    EXPECT_EQ(ir.output_index, 1);
    EXPECT_NO_THROW(verify(ir));
}

TEST(Canonicalize, KeepsNonIdentityView) {
    IR ir;
    ir.nodes = {
        make_input({4}, {1}),
        make_layout(OpCode::VIEW, 0, {3}, {1}, /*offset=*/1),
        make_add(1, 1, {3}, {1}),
    };
    ir.output_index = 2;

    optimize_graph(ir);

    ASSERT_EQ(ir.nodes.size(), 3u);
    EXPECT_EQ(ir.nodes[1].op, OpCode::VIEW);
    EXPECT_EQ(ir.nodes[2].inputs, (std::vector<int>{1, 1}));
    EXPECT_NO_THROW(verify(ir));
}

TEST(Canonicalize, SkipsIdentityChain) {
    IR ir;
    ir.nodes = {
        make_input({2}, {1}),
        make_layout(OpCode::VIEW, 0, {2}, {1}),
        make_layout(OpCode::TRANSPOSE, 1, {2}, {1}),
        make_add(2, 2, {2}, {1}),
    };
    ir.output_index = 3;

    optimize_graph(ir);

    ASSERT_EQ(ir.nodes.size(), 2u);
    EXPECT_EQ(ir.nodes[1].op, OpCode::ADD);
    EXPECT_EQ(ir.nodes[1].inputs, (std::vector<int>{0, 0}));
    EXPECT_EQ(ir.output_index, 1);
    EXPECT_NO_THROW(verify(ir));
}
