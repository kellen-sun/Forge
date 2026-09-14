#include <cmath>

#include "../include/compiler.h"
#include "../include/pass.h"

static bool is_constant(const IR& ir, int index) {
    return ir.nodes[index].op == OpCode::CONSTANT && ir.nodes[index].args.size() == 1;
}

static float fold_unary(int64_t kind, float value) {
    switch (kind) {
        case 0:
            return std::exp(value);
        case 1:
            return std::exp2(value);
        case 2:
            return std::pow(10.0f, value);
        case 3:
            return std::log(value);
        case 4:
            return std::log2(value);
        case 5:
            return std::log10(value);
        case 6:
            return std::sqrt(value);
        case 7:
            return 1.0f / std::sqrt(value);
        case 8:
            return std::fabs(value);
        case 9:
            return (value > 0.0f) - (value < 0.0f);
        case 10:
            return std::ceil(value);
        case 11:
            return std::floor(value);
        case 12:
            return std::round(value);
        case 13:
            return std::trunc(value);
        case 14:
            return value - std::floor(value);
        case 15:
            return std::sin(value);
        case 16:
            return std::cos(value);
        case 17:
            return std::tan(value);
        case 18:
            return std::asin(value);
        case 19:
            return std::acos(value);
        case 20:
            return std::atan(value);
        case 21:
            return std::sinh(value);
        case 22:
            return std::cosh(value);
        case 23:
            return std::tanh(value);
        default:
            return value;
    }
}

static bool fold_node(IR& ir, int index) {
    Node& node = ir.nodes[index];
    float result = 0.0f;

    if (node.op == OpCode::UNARY && node.inputs.size() == 1 && node.args.size() == 1 &&
        is_constant(ir, node.inputs[0])) {
        result = fold_unary(node.args[0], decode_f32(ir.nodes[node.inputs[0]].args[0]));
    } else if ((node.op == OpCode::ADD || node.op == OpCode::SUB ||
                node.op == OpCode::MUL || node.op == OpCode::DIV) &&
               node.inputs.size() == 2 && is_constant(ir, node.inputs[0]) &&
               is_constant(ir, node.inputs[1])) {
        const float lhs = decode_f32(ir.nodes[node.inputs[0]].args[0]);
        const float rhs = decode_f32(ir.nodes[node.inputs[1]].args[0]);
        switch (node.op) {
            case OpCode::ADD:
                result = lhs + rhs;
                break;
            case OpCode::SUB:
                result = lhs - rhs;
                break;
            case OpCode::MUL:
                result = lhs * rhs;
                break;
            case OpCode::DIV:
                result = lhs / rhs;
                break;
            default:
                return false;
        }
    } else {
        return false;
    }

    node.op = OpCode::CONSTANT;
    node.inputs.clear();
    node.args = {encode_f32(result)};
    return true;
}

void ConstantFoldPass::run(IR& ir) {
    for (int i = 0; i < static_cast<int>(ir.nodes.size()); ++i) {
        fold_node(ir, i);
    }
}
