#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "../../cpp/include/ir.h"
#include "../../cpp/include/pass.h"
#include "ir_builders.h"

TEST(Verify, AcceptsWellFormedIR) {
    IR ir;
    ir.nodes = {make_input({2, 2}, {2, 1}), make_input({2, 2}, {2, 1}),
                make_add(0, 1, {2, 2}, {2, 1})};
    ir.output_index = 2;
    EXPECT_NO_THROW(verify(ir));
}

TEST(Verify, RejectsOutOfRangeOperand) {
    IR ir;
    ir.nodes = {make_input({2, 2}, {2, 1}), make_add(0, 99, {2, 2}, {2, 1})};
    ir.output_index = 1;
    EXPECT_THROW(verify(ir), std::runtime_error);
}

TEST(Verify, RejectsWrongArity) {
    IR ir;
    Node add = make_add(0, 0, {2, 2}, {2, 1});
    add.inputs = {0};
    ir.nodes = {make_input({2, 2}, {2, 1}), add};
    ir.output_index = 1;
    EXPECT_THROW(verify(ir), std::runtime_error);
}

TEST(Verify, PassManagerPrefixesPassNameOnFailure) {
    struct BreaksIR : Pass {
        const char* name() const override { return "breaks-ir"; }
        void run(IR& ir) override { ir.nodes[0].inputs = {99}; }
    };

    IR ir;
    ir.nodes = {make_input({2}, {1})};
    ir.output_index = 0;

    PassManager pm;
    pm.addPass(std::make_unique<BreaksIR>());
    try {
        pm.run(ir);
        FAIL() << "expected verify to throw after pass";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("breaks-ir"), std::string::npos);
    }
}
