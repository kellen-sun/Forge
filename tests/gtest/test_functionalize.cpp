#include <gtest/gtest.h>

#include "../../cpp/include/compiler.h"
#include "../../cpp/include/ir.h"
#include "../../cpp/include/pass.h"
#include "ir_builders.h"

static Node constant(float value) {
    return {OpCode::CONSTANT, {}, {1}, {1}, 0, {encode_f32(value)}};
}

// x[1] = value, as traced on a destination with the given layout
static Node update_index_1(int dest, int value, std::vector<int64_t> shape,
                           std::vector<int64_t> strides) {
    return {OpCode::UPDATE, {dest, value}, std::move(shape), std::move(strides), 0,
            {1, 1, 1, 0}};
}

TEST(Functionalize, LeavesGraphWithoutUpdatesAlone) {
    IR ir;
    ir.nodes = {
        make_input({2}, {1}),
        make_layout(OpCode::VIEW, 0, {1}, {1}, 1),
        make_add(1, 1, {1}, {1}),
    };
    ir.output_index = 2;

    FunctionalizePass{}.run(ir);

    ASSERT_EQ(ir.nodes.size(), 3u);
    EXPECT_EQ(ir.nodes[2].inputs, (std::vector<int>{1, 1}));
    EXPECT_EQ(ir.output_index, 2);
}

TEST(Functionalize, ReplaysViewTakenBeforeWriteOnNewVersion) {
    // v = x[0:2]; x[1] = c; return v + c
    IR ir;
    ir.nodes = {
        make_input({3}, {1}),
        make_layout(OpCode::VIEW, 0, {2}, {1}),
        constant(5.0f),
        update_index_1(0, 2, {3}, {1}),
        make_add(1, 2, {2}, {1}),
    };
    ir.output_index = 4;

    FunctionalizePass{}.run(ir);

    ASSERT_EQ(ir.nodes.size(), 6u);
    EXPECT_EQ(ir.nodes[4].op, OpCode::VIEW);
    EXPECT_EQ(ir.nodes[4].inputs, (std::vector<int>{3}));
    EXPECT_EQ(ir.nodes[4].shape, (std::vector<int64_t>{2}));
    EXPECT_EQ(ir.nodes[5].inputs, (std::vector<int>{4, 2}));
    EXPECT_EQ(ir.output_index, 5);
    EXPECT_NO_THROW(verify(ir));
}

TEST(Functionalize, WidensUpdateThroughViewToWholeRoot) {
    // v = x[0:2]; v[1] = c; return x + c
    IR ir;
    ir.nodes = {
        make_input({3}, {1}),
        make_layout(OpCode::VIEW, 0, {2}, {1}),
        constant(5.0f),
        update_index_1(1, 2, {2}, {1}),
        make_add(0, 2, {3}, {1}),
    };
    ir.output_index = 4;

    FunctionalizePass{}.run(ir);

    ASSERT_EQ(ir.nodes.size(), 5u);
    EXPECT_EQ(ir.nodes[3].inputs, (std::vector<int>{0, 2}));
    EXPECT_EQ(ir.nodes[3].shape, (std::vector<int64_t>{3}));
    EXPECT_EQ(ir.nodes[4].inputs, (std::vector<int>{3, 2}));
    EXPECT_NO_THROW(verify(ir));
}

TEST(Functionalize, KeepsViewShapeForReadsOfUpdateThroughView) {
    // v = x[0:2]; v[1] = c; return v
    IR ir;
    ir.nodes = {
        make_input({3}, {1}),
        make_layout(OpCode::VIEW, 0, {2}, {1}),
        constant(5.0f),
        update_index_1(1, 2, {2}, {1}),
    };
    ir.output_index = 3;

    FunctionalizePass{}.run(ir);

    ASSERT_EQ(ir.nodes.size(), 5u);
    EXPECT_EQ(ir.nodes[4].op, OpCode::VIEW);
    EXPECT_EQ(ir.nodes[4].inputs, (std::vector<int>{3}));
    EXPECT_EQ(ir.nodes[4].shape, (std::vector<int64_t>{2}));
    EXPECT_EQ(ir.output_index, 4);
    EXPECT_NO_THROW(verify(ir));
}

TEST(SnapshotStaleReads, LeavesReadsBeforeWriteAlone) {
    IR ir;
    ir.nodes = {
        make_input({3}, {1}),
        constant(5.0f),
        make_add(0, 1, {3}, {1}),
        update_index_1(0, 1, {3}, {1}),
    };
    ir.output_index = 2;

    SnapshotStaleReadsPass{}.run(ir);

    ASSERT_EQ(ir.nodes.size(), 4u);
    EXPECT_EQ(ir.nodes[2].inputs, (std::vector<int>{0, 1}));
}

TEST(SnapshotStaleReads, CopiesVersionReadAfterItIsOverwritten) {
    // the ADD reads x0 after the UPDATE has overwritten it in place
    IR ir;
    ir.nodes = {
        make_input({3}, {1}),
        constant(5.0f),
        update_index_1(0, 1, {3}, {1}),
        make_add(0, 1, {3}, {1}),
    };
    ir.output_index = 3;

    SnapshotStaleReadsPass{}.run(ir);

    ASSERT_EQ(ir.nodes.size(), 5u);
    EXPECT_EQ(ir.nodes[2].op, OpCode::COPY);
    EXPECT_EQ(ir.nodes[2].inputs, (std::vector<int>{0}));
    EXPECT_EQ(ir.nodes[3].op, OpCode::UPDATE);
    EXPECT_EQ(ir.nodes[3].inputs, (std::vector<int>{0, 1}));
    EXPECT_EQ(ir.nodes[4].inputs, (std::vector<int>{2, 1}));
    EXPECT_EQ(ir.output_index, 4);
    EXPECT_NO_THROW(verify(ir));
}
