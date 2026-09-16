#include <gtest/gtest.h>

#include <utility>

#include "../../cpp/include/graph_builder.h"
#include "../../cpp/include/pass.h"
#include "ir_builders.h"

TEST(GraphBuilder, AppendsAndRewritesUses) {
    IR ir;
    ir.nodes = {
        make_input({2}, {1}),
        make_input({2}, {1}),
    };
    ir.output_index = 0;

    GraphBuilder builder(ir);
    const int add = builder.append(make_add(0, 1, {2}, {1}));
    Node consumer = make_add(add, 0, {2}, {1});
    const int consumer_id = builder.append(std::move(consumer));
    builder.replace(add, 0);

    EXPECT_EQ(ir.nodes[consumer_id].inputs, (std::vector<int>{0, 0}));
    EXPECT_EQ(ir.output_index, 0);
    EXPECT_NO_THROW(verify(ir));
}

TEST(GraphBuilder, RejectsForwardOperand) {
    IR ir;
    ir.nodes = {make_input({2}, {1})};
    GraphBuilder builder(ir);
    Node invalid = make_add(0, 3, {2}, {1});
    EXPECT_THROW(builder.append(std::move(invalid)), std::runtime_error);
}
