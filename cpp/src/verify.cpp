#include <stdexcept>
#include <string>

#include "../include/fused_elementwise.h"
#include "../include/pass.h"

static constexpr int kUnknownArity = -1;
static constexpr int kVariableArity = -2;

static int expected_arity(OpCode op) {
    switch (op) {
        case OpCode::INPUT:
        case OpCode::CONSTANT:
        case OpCode::ZEROS:
        case OpCode::RAND:
        case OpCode::RANDN:
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
        case OpCode::FUSED_ELEMENTWISE:
            return kVariableArity;
        case OpCode::COUNT:
            return kUnknownArity;
    }
    return kUnknownArity;
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
        if (arity == kUnknownArity) {
            throw std::runtime_error("verify: unknown opcode at node " + std::to_string(i));
        }
        if (arity >= 0 && static_cast<int>(node.inputs.size()) != arity) {
            throw std::runtime_error("verify: node " + std::to_string(i) + " has wrong arity");
        }
        if (node.op == OpCode::FUSED_ELEMENTWISE) {
            if (node.args.size() < kFusedHeaderSize ||
                node.args[0] != kFusedEncodingVersion || node.args[1] < 1 ||
                node.args.size() !=
                    kFusedHeaderSize +
                        static_cast<size_t>(node.args[1]) * kFusedInstructionWidth) {
                throw std::runtime_error("verify: malformed FUSED_ELEMENTWISE encoding");
            }
            const int64_t instruction_count = node.args[1];
            const auto valid_ref = [&](int64_t ref, int64_t instruction) {
                if (ref >= 0) return ref < static_cast<int64_t>(node.inputs.size());
                const int64_t temp = -1 - ref;
                return temp >= 0 && temp < instruction;
            };
            for (int64_t instruction = 0; instruction < instruction_count; ++instruction) {
                const size_t base = kFusedHeaderSize +
                                    static_cast<size_t>(instruction) * kFusedInstructionWidth;
                const int64_t op = node.args[base];
                if (op < static_cast<int64_t>(FusedOp::ADD) ||
                    op > static_cast<int64_t>(FusedOp::UNARY) ||
                    !valid_ref(node.args[base + 1], instruction) ||
                    (op != static_cast<int64_t>(FusedOp::UNARY) &&
                     !valid_ref(node.args[base + 2], instruction))) {
                    throw std::runtime_error("verify: malformed FUSED_ELEMENTWISE instruction");
                }
                if (op == static_cast<int64_t>(FusedOp::UNARY) &&
                    (node.args[base + 3] < 0 || node.args[base + 3] >= kUnaryCount)) {
                    throw std::runtime_error("verify: invalid fused unary kind");
                }
            }
            if (!valid_ref(node.args[2], instruction_count)) {
                throw std::runtime_error("verify: invalid FUSED_ELEMENTWISE output");
            }
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
