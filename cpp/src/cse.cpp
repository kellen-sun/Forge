#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "../include/compiler.h"
#include "../include/ir_utils.h"
#include "../include/pass.h"

static bool is_pure_elementwise(OpCode op) {
    return op == OpCode::CONSTANT || op == OpCode::ADD || op == OpCode::SUB ||
           op == OpCode::MUL || op == OpCode::DIV || op == OpCode::UNARY;
}

static bool same_layout(const Node& a, const Node& b) {
    return a.shape == b.shape && a.strides == b.strides && a.offset == b.offset;
}

static bool is_singleton_constant(const IR& ir, int index, float value) {
    const Node& node = ir.nodes[index];
    return node.op == OpCode::CONSTANT && node.args.size() == 1 &&
           decode_f32(node.args[0]) == value;
}

static int identity_source(const IR& ir, const Node& node) {
    if (node.inputs.size() != 2) return -1;
    const int lhs_index = node.inputs[0];
    const int rhs_index = node.inputs[1];
    if (!same_layout(node, ir.nodes[lhs_index])) return -1;

    const bool additive_identity =
        (node.op == OpCode::ADD || node.op == OpCode::SUB) &&
        is_singleton_constant(ir, rhs_index, 0.0f);
    const bool multiplicative_identity =
        (node.op == OpCode::MUL || node.op == OpCode::DIV) &&
        is_singleton_constant(ir, rhs_index, 1.0f);
    return (additive_identity || multiplicative_identity) ? lhs_index : -1;
}

static std::string cse_key(const Node& node) {
    std::ostringstream key;
    key << static_cast<int>(node.op) << ':' << node.offset << ':';
    for (int input : node.inputs) key << input << ',';
    key << '|';
    for (int64_t arg : node.args) key << arg << ',';
    key << '|';
    for (int64_t dim : node.shape) key << dim << ',';
    key << '|';
    for (int64_t stride : node.strides) key << stride << ',';
    return key.str();
}

void CSEPass::run(IR& ir) {
    std::unordered_map<std::string, int> available;
    std::unordered_set<int> mutated_roots;
    std::vector<int> remap(ir.nodes.size());
    for (int i = 0; i < static_cast<int>(ir.nodes.size()); ++i) remap[i] = i;

    for (const Node& node : ir.nodes) {
        if (node.op == OpCode::UPDATE && !node.inputs.empty()) {
            mutated_roots.insert(storage_root(ir.nodes, node.inputs[0]));
        }
    }

    for (int i = 0; i < static_cast<int>(ir.nodes.size()); ++i) {
        Node& node = ir.nodes[i];
        for (int& input : node.inputs) input = remap[input];

        if (is_pure_elementwise(node.op)) {
            const int source = identity_source(ir, node);
            if (source >= 0 && !mutated_roots.count(storage_root(ir.nodes, source))) {
                remap[i] = source;
                continue;
            }
        }

        if (!is_pure_elementwise(node.op) ||
            mutated_roots.count(storage_root(ir.nodes, i))) {
            continue;
        }

        const std::string key = cse_key(node);
        auto [it, inserted] = available.emplace(key, i);
        if (!inserted) remap[i] = it->second;
    }

    ir.output_index = remap[ir.output_index];
    for (Node& node : ir.nodes) {
        for (int& input : node.inputs) input = remap[input];
    }
}
