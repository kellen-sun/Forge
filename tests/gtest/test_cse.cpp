#include <gtest/gtest.h>

#include "../../cpp/include/compiler.h"
#include "../../cpp/include/pass.h"
#include "ir_builders.h"

static Node constant(float value) {
    Node n;
    n.op = OpCode::CONSTANT;
    n.shape = {1};
    n.strides = {1};
    n.offset = 0;
    n.args = {encode_f32(value)};
    return n;
}

TEST(CSE, RemapsDuplicatePureExpression) {
    IR ir;
    ir.nodes = {
        make_input({2}, {1}),
        make_input({2}, {1}),
        make_add(0, 1, {2}, {1}),
        make_add(0, 1, {2}, {1}),
    };
    ir.output_index = 3;

    CSEPass{}.run(ir);

    EXPECT_EQ(ir.output_index, 2);
    EXPECT_EQ(ir.nodes[3].inputs, (std::vector<int>{0, 1}));
    EXPECT_NO_THROW(verify(ir));
}

TEST(CSE, SimplifiesSafeIdentityConstants) {
    IR ir;
    Node zero = constant(0.0f);
    Node one = constant(1.0f);
    ir.nodes = {
        make_input({2}, {1}),
        zero,
        make_add(0, 1, {2}, {1}),
        one,
        {OpCode::MUL, {2, 3}, {2}, {1}, 0, {}},
    };
    ir.output_index = 4;

    CSEPass{}.run(ir);

    EXPECT_EQ(ir.output_index, 0);
    EXPECT_NO_THROW(verify(ir));
}

TEST(CSE, DoesNotMergeValueThatIsLaterMutated) {
    IR ir;
    ir.nodes = {
        make_input({2}, {1}),
        make_input({2}, {1}),
        make_add(0, 1, {2}, {1}),
        make_add(0, 1, {2}, {1}),
        constant(1.0f),
        {OpCode::UPDATE, {3, 4}, {2}, {1}, 0, {2, 1, 0, 0}},
    };
    ir.output_index = 2;

    CSEPass{}.run(ir);

    EXPECT_EQ(ir.nodes[2].inputs, (std::vector<int>{0, 1}));
    EXPECT_EQ(ir.nodes[3].inputs, (std::vector<int>{0, 1}));
    EXPECT_EQ(ir.nodes[5].inputs[0], 3);
    EXPECT_EQ(ir.output_index, 2);
    EXPECT_NO_THROW(verify(ir));
}

TEST(CSE, AppliesFastMathCommutativityAndZeroProduct) {
    Node zero = constant(0.0f);
    IR ir;
    ir.nodes = {
        make_input({2}, {1}),
        make_input({2}, {1}),
        make_add(1, 0, {2}, {1}),
        zero,
        {OpCode::MUL, {0, 3}, {2}, {1}, 0, {}},
    };
    ir.output_index = 2;

    CSEPass{}.run(ir);

    EXPECT_EQ(ir.output_index, 2);
    EXPECT_EQ(ir.nodes[2].inputs, (std::vector<int>{1, 0}));
    EXPECT_EQ(ir.nodes[4].op, OpCode::CONSTANT);
    EXPECT_FLOAT_EQ(decode_f32(ir.nodes[4].args[0]), 0.0f);
    EXPECT_NO_THROW(verify(ir));
}
