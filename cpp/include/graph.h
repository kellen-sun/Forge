#pragma once
#include <memory>
#include <string>
#include <vector>

#include "array_handle.h"
#include "common.h"

struct Node {
    OpCode op;
    std::vector<int> inputs;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    int64_t offset;

    // Flatten args
    std::vector<int64_t> args;
};

struct KernelConfig {
    std::string name;             // e.g., "op_3_add"
    std::vector<uint64_t> grid;   // Global Dispatch Size (e.g., [1024, 1, 1])
    std::vector<uint64_t> group;  // Local Workgroup Size (e.g., [256, 1, 1])
};

class MemoryArena;

class Graph {
   public:
    std::vector<Node> nodes;
    int output_index;

    std::shared_ptr<MemoryArena> arena;

    // Metal Source Code
    std::string shader_source;
    // One config per graph node
    std::vector<KernelConfig> configs;

    // CONSTRUCTORS //
    Graph(std::vector<Node> nodes, int output_index)
        : nodes(std::move(nodes)), output_index(output_index) {}

    std::shared_ptr<ArrayHandle> execute(std::vector<std::shared_ptr<ArrayHandle>> inputs);
};
