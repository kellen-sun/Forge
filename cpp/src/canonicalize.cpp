#include "../include/ir_utils.h"
#include "../include/pass.h"

static bool is_identity_layout(const Node& node, const Node& src) {
    return is_layout_op(node.op) && node.inputs.size() == 1 && node.shape == src.shape &&
           node.strides == src.strides && node.offset == src.offset;
}

static int skip_identity(const IR& ir, int i) {
    while (i >= 0 && i < static_cast<int>(ir.nodes.size())) {
        const Node& node = ir.nodes[i];
        if (node.inputs.size() != 1) break;
        const Node& src = ir.nodes[node.inputs[0]];
        if (!is_identity_layout(node, src)) break;
        i = node.inputs[0];
    }
    return i;
}

void CanonicalizePass::run(IR& ir) {
    for (auto& node : ir.nodes) {
        for (int& inp : node.inputs) inp = skip_identity(ir, inp);
    }
    ir.output_index = skip_identity(ir, ir.output_index);
}
