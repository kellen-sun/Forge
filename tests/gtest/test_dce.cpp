#include <gtest/gtest.h>

#include "../../cpp/include/compiler.h"
#include "../../cpp/include/ir.h"
#include "../../cpp/include/pass.h"
#include "ir_builders.h"

TEST(DCE, DropsUnreachableComputeKeepsInputs) {
    IR ir;
    ir.nodes = {
        make_input({2}, {1}),
        make_input({2}, {1}),
        make_add(0, 1, {2}, {1}),
        make_add(0, 1, {2}, {1}),
    };
    ir.output_index = 3;

    DCEPass{}.run(ir);

    ASSERT_EQ(ir.nodes.size(), 3u);
    EXPECT_EQ(ir.nodes[0].op, OpCode::INPUT);
    EXPECT_EQ(ir.nodes[1].op, OpCode::INPUT);
    EXPECT_EQ(ir.nodes[2].op, OpCode::ADD);
    EXPECT_EQ(ir.nodes[2].inputs, (std::vector<int>{0, 1}));
    EXPECT_EQ(ir.output_index, 2);
    EXPECT_NO_THROW(verify(ir));
}

TEST(DCE, KeepsUnusedInput) {
    Node c;
    c.op = OpCode::CONSTANT;
    c.shape = {1};
    c.strides = {1};
    c.offset = 0;
    c.args = {encode_f32(1.0f)};

    IR ir;
    ir.nodes = {
        make_input({2}, {1}),
        make_input({2}, {1}),
        c,
        make_add(0, 2, {2}, {1}),
    };
    ir.output_index = 3;

    DCEPass{}.run(ir);

    ASSERT_EQ(ir.nodes.size(), 4u);
    EXPECT_EQ(ir.nodes[1].op, OpCode::INPUT);
    EXPECT_EQ(ir.output_index, 3);
    EXPECT_EQ(ir.nodes[3].inputs, (std::vector<int>{0, 2}));
    EXPECT_NO_THROW(verify(ir));
}
