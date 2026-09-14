#include <array>
#include <stdexcept>
#include <string>

#include "../include/array_handle.h"
#include "../include/pass.h"

static void keep(IR&, int) {}

static const Node& operand(const IR& ir, const Node& node, int i) {
    return ir.nodes[node.inputs[i]];
}

static void infer_binary(IR& ir, int i) {
    Node& node = ir.nodes[i];
    const Node& a = operand(ir, node, 0);
    const Node& b = operand(ir, node, 1);
    node.shape = broadcast_shapes(a.shape, b.shape);
    node.strides = make_strides(node.shape);
}

static void infer_copy(IR& ir, int i) {
    Node& node = ir.nodes[i];
    node.shape = operand(ir, node, 0).shape;
    node.strides = make_strides(node.shape);
}

static void infer_reshape(IR& ir, int i) {
    Node& node = ir.nodes[i];
    const Node& src = operand(ir, node, 0);
    if (numel_from_shape(node.shape) != numel_from_shape(src.shape)) {
        throw std::runtime_error("shape-infer: RESHAPE numel mismatch at node " +
                                 std::to_string(i));
    }
    node.strides = make_strides(node.shape);
}

static void infer_update(IR& ir, int i) {
    Node& node = ir.nodes[i];
    const Node& dest = operand(ir, node, 0);
    node.shape = dest.shape;
    node.strides = dest.strides;
}

static void infer_matmul(IR& ir, int i) {
    Node& node = ir.nodes[i];
    node.shape = matmul_output_shape(operand(ir, node, 0).shape, operand(ir, node, 1).shape);
    node.strides = make_strides(node.shape);
}

// Indexed by OpCode
using InferFn = void (*)(IR&, int);
static constexpr int kOpCount = static_cast<int>(OpCode::COUNT);

static const std::array<InferFn, kOpCount> kInfer = [] {
    std::array<InferFn, kOpCount> t{};
    t[static_cast<int>(OpCode::INPUT)] = keep;
    t[static_cast<int>(OpCode::CONSTANT)] = keep;
    t[static_cast<int>(OpCode::VIEW)] = keep;
    t[static_cast<int>(OpCode::TRANSPOSE)] = keep;
    t[static_cast<int>(OpCode::ADD)] = infer_binary;
    t[static_cast<int>(OpCode::SUB)] = infer_binary;
    t[static_cast<int>(OpCode::MUL)] = infer_binary;
    t[static_cast<int>(OpCode::DIV)] = infer_binary;
    t[static_cast<int>(OpCode::COPY)] = infer_copy;
    t[static_cast<int>(OpCode::RESHAPE)] = infer_reshape;
    t[static_cast<int>(OpCode::UPDATE)] = infer_update;
    t[static_cast<int>(OpCode::MATMUL)] = infer_matmul;
    return t;
}();

void ShapeInferPass::run(IR& ir) {
    for (int i = 0; i < static_cast<int>(ir.nodes.size()); ++i) {
        const int op = static_cast<int>(ir.nodes[i].op);
        if (op < 0 || op >= kOpCount || kInfer[op] == nullptr) {
            throw std::runtime_error("shape-infer: no inference for opcode " +
                                     std::to_string(op));
        }
        kInfer[op](ir, i);
    }
}
