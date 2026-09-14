#include <gtest/gtest.h>

#include "../../cpp/include/compiler.h"
#include "../../cpp/include/ir.h"
#include "../../cpp/include/pass.h"
#include "ir_builders.h"

static Node make_update(int dest, int value, std::vector<int64_t> shape,
                        std::vector<int64_t> strides, std::vector<int64_t> target_args) {
    Node n;
    n.op = OpCode::UPDATE;
    n.inputs = {dest, value};
    n.shape = std::move(shape);
    n.strides = std::move(strides);
    n.offset = 0;
    n.args = std::move(target_args);
    return n;
}

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

TEST(DCE, KeepsInputRootedUpdateForSideEffect) {
    Node value;
    value.op = OpCode::CONSTANT;
    value.shape = {1};
    value.strides = {1};
    value.offset = 0;
    value.args = {encode_f32(7.0f)};

    IR ir;
    ir.nodes = {
        make_input({2}, {1}),
        value,
        make_update(0, 1, {2}, {1}, {1, 1, 0}),
    };
    ir.output_index = 0;

    DCEPass{}.run(ir);

    ASSERT_EQ(ir.nodes.size(), 3u);
    EXPECT_EQ(ir.nodes[2].op, OpCode::UPDATE);
    EXPECT_EQ(ir.output_index, 0);
    EXPECT_NO_THROW(verify(ir));
}

TEST(DCE, DropsUpdateToDeadTemporary) {
    Node value;
    value.op = OpCode::CONSTANT;
    value.shape = {1};
    value.strides = {1};
    value.offset = 0;
    value.args = {encode_f32(7.0f)};

    IR ir;
    ir.nodes = {
        make_input({2}, {1}),
        value,
        make_add(0, 1, {2}, {1}),
        make_update(2, 1, {2}, {1}, {1, 1, 0}),
    };
    ir.output_index = 0;

    DCEPass{}.run(ir);

    ASSERT_EQ(ir.nodes.size(), 1u);
    EXPECT_EQ(ir.nodes[0].op, OpCode::INPUT);
    EXPECT_EQ(ir.output_index, 0);
    EXPECT_NO_THROW(verify(ir));
}
