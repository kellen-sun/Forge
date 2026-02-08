#pragma once
#include <memory>
#include <vector>

#include "array_handle.h"
#include "common.h"

struct Node {
    OpCode op;
    std::vector<int> inputs;
    std::vector<uint64_t> shape;
    std::vector<uint64_t> strides;
    uint64_t offset;

    // Flatten args
    std::vector<uint64_t> args;
};

class Graph {
   public:
    std::vector<Node> nodes;
    int output_index;

    Graph(std::vector<Node> nodes, int output_index)
        : nodes(std::move(nodes)), output_index(output_index) {}

    std::shared_ptr<ArrayHandle> execute(std::vector<std::shared_ptr<ArrayHandle>> inputs);
};
