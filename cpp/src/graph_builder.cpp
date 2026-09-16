#include <stdexcept>
#include <utility>

#include "../include/graph_builder.h"

int GraphBuilder::append(Node node) {
    const int new_id = static_cast<int>(ir_.nodes.size());
    for (int input : node.inputs) {
        if (input < 0 || input >= new_id) {
            throw std::runtime_error("GraphBuilder: appended node has invalid input");
        }
    }
    ir_.nodes.push_back(std::move(node));
    return new_id;
}

void GraphBuilder::replaceUses(int old_id, int new_id) {
    if (old_id < 0 || old_id >= static_cast<int>(ir_.nodes.size()) || new_id < 0 ||
        new_id >= static_cast<int>(ir_.nodes.size())) {
        throw std::runtime_error("GraphBuilder: replacement ID out of range");
    }
    for (Node& node : ir_.nodes) {
        for (int& input : node.inputs) {
            if (input == old_id) input = new_id;
        }
    }
}

void GraphBuilder::replace(int old_id, int new_id) {
    replaceUses(old_id, new_id);
    if (ir_.output_index == old_id) ir_.output_index = new_id;
}

void GraphBuilder::setOutput(int node_id) {
    if (node_id < 0 || node_id >= static_cast<int>(ir_.nodes.size())) {
        throw std::runtime_error("GraphBuilder: output ID out of range");
    }
    ir_.output_index = node_id;
}
