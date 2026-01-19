#include "../include/graph_handle.h"
#include "../include/array_binops.h"

#include <stdexcept>
#include <string>

std::shared_ptr<GraphHandle> make_graph_from_flat(
    const std::vector<std::tuple<int, std::vector<int>, std::vector<int64_t>, int64_t, std::vector<int64_t>>>& flat_nodes,
    int output_idx) {

    std::vector<GraphNode> nodes;
    nodes.reserve(flat_nodes.size());

    for (const auto& [op_int, inputs, shape, offset, strides] : flat_nodes) {
        GraphNode node;
        node.op = static_cast<OpCode>(op_int);
        node.inputs = inputs;
        node.shape = shape;
        node.offset = offset;
        node.strides = strides;
        nodes.push_back(std::move(node));
    }

    return std::make_shared<GraphHandle>(std::move(nodes), output_idx);
}

std::shared_ptr<ArrayHandle> GraphHandle::execute(
    const std::vector<std::shared_ptr<ArrayHandle>>& inputs) {

    // Values array: holds ArrayHandle for each node
    std::vector<std::shared_ptr<ArrayHandle>> values(nodes_.size());

    // Counter for INPUT nodes
    size_t input_counter = 0;

    // Execute nodes in order (they're already topologically sorted)
    for (size_t i = 0; i < nodes_.size(); ++i) {
        const GraphNode& node = nodes_[i];

        switch (node.op) {
            case OpCode::INPUT: {
                // Get the next input from the inputs array
                if (input_counter >= inputs.size()) {
                    throw std::runtime_error(
                        "GraphHandle::execute: not enough inputs provided. "
                        "Expected at least " + std::to_string(input_counter + 1) +
                        ", got " + std::to_string(inputs.size()));
                }
                values[i] = inputs[input_counter++];
                break;
            }

            case OpCode::ADD: {
                auto& lhs = values[node.inputs[0]];
                auto& rhs = values[node.inputs[1]];
                values[i] = array_binops(lhs, rhs, "add");
                break;
            }

            case OpCode::SUB: {
                auto& lhs = values[node.inputs[0]];
                auto& rhs = values[node.inputs[1]];
                values[i] = array_binops(lhs, rhs, "sub");
                break;
            }

            case OpCode::MUL: {
                auto& lhs = values[node.inputs[0]];
                auto& rhs = values[node.inputs[1]];
                values[i] = array_binops(lhs, rhs, "mul");
                break;
            }

            case OpCode::DIV: {
                auto& lhs = values[node.inputs[0]];
                auto& rhs = values[node.inputs[1]];
                values[i] = array_binops(lhs, rhs, "div");
                break;
            }

            case OpCode::MATMUL: {
                auto& lhs = values[node.inputs[0]];
                auto& rhs = values[node.inputs[1]];
                values[i] = array_matmul(lhs, rhs);
                break;
            }

            default:
                throw std::runtime_error(
                    "GraphHandle::execute: unknown op code " +
                    std::to_string(static_cast<int>(node.op)));
        }
    }

    // Return the output node
    if (output_idx_ < 0 || output_idx_ >= static_cast<int>(values.size())) {
        throw std::runtime_error(
            "GraphHandle::execute: invalid output index " +
            std::to_string(output_idx_));
    }

    return values[output_idx_];
}
