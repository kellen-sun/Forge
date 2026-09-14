#include <gtest/gtest.h>

#include "../../cpp/include/compiler.h"
#include "../../cpp/include/ir.h"
#include "../../cpp/include/pass.h"
#include "ir_builders.h"

TEST(ShapeInfer, BroadcastAddOverwritesWrongShape) {
    IR ir;
    ir.nodes = {
        make_input({2, 1}, {1, 1}),
        make_input({1, 3}, {3, 1}),
        make_add(0, 1, {99}, {1}),
    };
    ir.output_index = 2;

    ShapeInferPass{}.run(ir);

    EXPECT_EQ(ir.nodes[2].shape, (std::vector<int64_t>{2, 3}));
    EXPECT_EQ(ir.nodes[2].strides, (std::vector<int64_t>{3, 1}));
    EXPECT_NO_THROW(verify(ir));
}

TEST(ShapeInfer, IdentityViewKeepsLayout) {
    IR ir;
    ir.nodes = {
        make_input({2, 2}, {2, 1}),
        make_layout(OpCode::VIEW, 0, {2, 2}, {2, 1}),
    };
    ir.output_index = 1;

    ShapeInferPass{}.run(ir);

    EXPECT_EQ(ir.nodes[1].op, OpCode::VIEW);
    EXPECT_EQ(ir.nodes[1].shape, (std::vector<int64_t>{2, 2}));
    EXPECT_EQ(ir.nodes[1].strides, (std::vector<int64_t>{2, 1}));
    EXPECT_NO_THROW(verify(ir));
}

TEST(ShapeInfer, ReshapeNumelMismatchThrows) {
    IR ir;
    ir.nodes = {
        make_input({2, 2}, {2, 1}),
        make_layout(OpCode::RESHAPE, 0, {3}, {1}),
    };
    ir.output_index = 1;
    EXPECT_THROW(ShapeInferPass{}.run(ir), std::runtime_error);
}

TEST(ShapeInfer, IncompatibleBroadcastThrows) {
    IR ir;
    ir.nodes = {
        make_input({2}, {1}),
        make_input({3}, {1}),
        make_add(0, 1, {2}, {1}),
    };
    ir.output_index = 2;
    EXPECT_THROW(ShapeInferPass{}.run(ir), std::runtime_error);
}

TEST(ShapeInfer, SumGlobalAndAxis) {
    Node gsum;
    gsum.op = OpCode::SUM;
    gsum.inputs = {0};
    gsum.args = {0};
    gsum.offset = 0;

    Node asize;
    asize.op = OpCode::SUM;
    asize.inputs = {0};
    asize.args = {1, 1};
    asize.offset = 0;

    IR ir;
    ir.nodes = {make_input({2, 3}, {3, 1}), gsum};
    ir.output_index = 1;
    ShapeInferPass{}.run(ir);
    EXPECT_TRUE(ir.nodes[1].shape.empty());
    EXPECT_TRUE(ir.nodes[1].strides.empty());

    ir.nodes = {make_input({2, 3}, {3, 1}), asize};
    ir.output_index = 1;
    ShapeInferPass{}.run(ir);
    EXPECT_EQ(ir.nodes[1].shape, (std::vector<int64_t>{2, 1}));
    EXPECT_EQ(ir.nodes[1].strides, (std::vector<int64_t>{1, 1}));
}

TEST(ShapeInfer, PipelineStillAcceptsTracedShapes) {
    IR ir;
    ir.nodes = {
        make_input({2, 2}, {2, 1}),
        make_input({2, 2}, {2, 1}),
        make_add(0, 1, {2, 2}, {2, 1}),
    };
    ir.output_index = 2;
    EXPECT_NO_THROW(optimize_graph(ir));
    EXPECT_EQ(ir.nodes[2].shape, (std::vector<int64_t>{2, 2}));
}
