#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "array_handle.h"

// Op codes - must match Python graph.py Ops class
enum class OpCode : int {
    INPUT = 0,
    MATMUL = 1,
    ADD = 2,
    MUL = 3,
    DIV = 4,
    SUB = 5,
};

struct GraphNode {
    OpCode op;
    std::vector<int> inputs;      // Indices into the values array
    std::vector<int64_t> shape;   // Output shape of this node
    int64_t offset;               // Offset into buffer (for views)
    std::vector<int64_t> strides; // Strides for this node
};

class GraphHandle {
   private:
    std::vector<GraphNode> nodes_;
    int output_idx_;

   public:
    GraphHandle(std::vector<GraphNode> nodes, int output_idx)
        : nodes_(std::move(nodes)), output_idx_(output_idx) {}

    // Execute the graph with concrete inputs
    // Returns the output ArrayHandle
    std::shared_ptr<ArrayHandle> execute(
        const std::vector<std::shared_ptr<ArrayHandle>>& inputs);

    // For debugging
    size_t num_nodes() const { return nodes_.size(); }
    int output_idx() const { return output_idx_; }
};

// Factory function to create GraphHandle from Python's flattened graph
// flat_graph is a tuple: (list of (op, input_ids, shape, offset, strides), output_idx)
std::shared_ptr<GraphHandle> make_graph_from_flat(
    const std::vector<std::tuple<int, std::vector<int>, std::vector<int64_t>, int64_t, std::vector<int64_t>>>& flat_nodes,
    int output_idx);
