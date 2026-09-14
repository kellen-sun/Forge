#include <gtest/gtest.h>

#include "../../cpp/include/compiler.h"
#include "../../cpp/include/pass.h"
#include "ir_builders.h"

static Node make_constant(float value, std::vector<int64_t> shape = {1},
                           std::vector<int64_t> strides = {1}) {
    Node n;
    n.op = OpCode::CONSTANT;
    n.shape = std::move(shape);
    n.strides = std::move(strides);
    n.offset = 0;
    n.args = {encode_f32(value)};
    return n;
}

TEST(ConstantFold, FoldsConstantBinary) {
    IR ir;
    ir.nodes = {
        make_constant(2.0f),
        make_constant(3.0f),
        make_add(0, 1, {2, 2}, {2, 1}),
    };
    ir.output_index = 2;

    ConstantFoldPass{}.run(ir);

    EXPECT_EQ(ir.nodes[2].op, OpCode::CONSTANT);
    EXPECT_TRUE(ir.nodes[2].inputs.empty());
    EXPECT_FLOAT_EQ(decode_f32(ir.nodes[2].args[0]), 5.0f);
    EXPECT_NO_THROW(verify(ir));
}

TEST(ConstantFold, FoldsConstantUnary) {
    Node unary;
    unary.op = OpCode::UNARY;
    unary.inputs = {0};
    unary.shape = {2};
    unary.strides = {1};
    unary.offset = 0;
    unary.args = {0};  // exp

    IR ir;
    ir.nodes = {make_constant(0.0f), unary};
    ir.output_index = 1;

    ConstantFoldPass{}.run(ir);

    EXPECT_EQ(ir.nodes[1].op, OpCode::CONSTANT);
    EXPECT_FLOAT_EQ(decode_f32(ir.nodes[1].args[0]), 1.0f);
    EXPECT_NO_THROW(verify(ir));
}
