#pragma once

#include <vector>

#include "graph.h"

inline bool is_layout_op(OpCode op) {
    return op == OpCode::VIEW || op == OpCode::RESHAPE || op == OpCode::TRANSPOSE;
}

inline bool is_storage_alias_op(OpCode op) {
    return is_layout_op(op) || op == OpCode::UPDATE;
}

inline int storage_root(const std::vector<Node>& nodes, int index) {
    while (index >= 0 && index < static_cast<int>(nodes.size())) {
        const Node& node = nodes[index];
        if (is_storage_alias_op(node.op) && !node.inputs.empty()) {
            index = node.inputs[0];
        } else {
            break;
        }
    }
    return index;
}
