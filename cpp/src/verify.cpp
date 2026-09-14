#include <stdexcept>
#include <string>

#include "../include/pass.h"

static int expected_arity(OpCode op) {
    switch (op) {
        case OpCode::INPUT:
        case OpCode::CONSTANT:
        case OpCode::ZEROS:
            return 0;
        case OpCode::RESHAPE:
        case OpCode::TRANSPOSE:
        case OpCode::VIEW:
        case OpCode::COPY:
        case OpCode::SUM:
        case OpCode::UNARY:
            return 1;
        case OpCode::MATMUL:
        case OpCode::ADD:
        case OpCode::MUL:
        case OpCode::DIV:
        case OpCode::SUB:
        case OpCode::UPDATE:
            return 2;
        case OpCode::COUNT:
            return -1;
    }
    return -1;
}

void verify(const IR& ir) {
    const int n = static_cast<int>(ir.nodes.size());
    if (n == 0) {
        throw std::runtime_error("verify: empty IR");
    }
    if (ir.output_index < 0 || ir.output_index >= n) {
        throw std::runtime_error("verify: output_index out of range");
    }

    for (int i = 0; i < n; ++i) {
        const Node& node = ir.nodes[i];
        const int arity = expected_arity(node.op);
        if (arity < 0) {
            throw std::runtime_error("verify: unknown opcode at node " + std::to_string(i));
        }
        if (static_cast<int>(node.inputs.size()) != arity) {
            throw std::runtime_error("verify: node " + std::to_string(i) + " has wrong arity");
        }
        if (node.shape.size() != node.strides.size()) {
            throw std::runtime_error("verify: node " + std::to_string(i) +
                                     " shape/strides rank mismatch");
        }
        if (node.op == OpCode::CONSTANT && node.args.empty()) {
            throw std::runtime_error("verify: CONSTANT node " + std::to_string(i) +
                                     " missing value");
        }
        if (node.op == OpCode::SUM && node.args.empty()) {
            throw std::runtime_error("verify: SUM node " + std::to_string(i) + " missing args");
        }
        if (node.op == OpCode::UNARY) {
            if (node.args.empty()) {
                throw std::runtime_error("verify: UNARY node " + std::to_string(i) +
                                         " missing kind");
            }
            const int64_t kind = node.args[0];
            if (kind < 0 || kind >= kUnaryCount) {
                throw std::runtime_error("verify: UNARY node " + std::to_string(i) +
                                         " has unknown kind");
            }
        }
        for (int inp : node.inputs) {
            if (inp < 0 || inp >= n) {
                throw std::runtime_error("verify: node " + std::to_string(i) +
                                         " has out-of-range operand");
            }
            if (inp >= i) {
                throw std::runtime_error("verify: node " + std::to_string(i) +
                                         " operand is not a prior value");
            }
        }
    }
}
