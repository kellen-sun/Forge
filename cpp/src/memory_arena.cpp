#include "../include/memory_arena.h"

MemoryArena::MemoryArena(const Graph& graph, uint64_t element_size) {
    // 1. Calculate array sizes and find roots of each array
    // ---> (root is the original array's memory being used in the case of a view, etc)
    size_t num_nodes = graph.nodes.size();
    // size required for each Array (the node) in bytes
    std::vector<int64_t> sizes(num_nodes);
    std::vector<int64_t> roots(num_nodes);

    for (size_t idx = 0; idx < num_nodes; ++idx) {
        OpCode op = graph.nodes[idx].op;
        roots[idx] = idx;
        if (op == OpCode::RESHAPE || op == OpCode::TRANSPOSE || op == OpCode::VIEW) {
            int parent = graph.nodes[idx].inputs[0];
            roots[idx] = roots[parent];
        }

        sizes[idx] = element_size * numel_from_shape(graph.nodes[idx].shape);
    }

    // 2. Set last used array, starting at -1, check when last seen inputs
    // Force the output Array's last used to be infinite, so it doesn't get recycled
    // 3. Simulate allocation, and frees using greedy best-fit
    // Walk through nodes in graph, if not enough memory available allocate more
    // Recycle dead memory, return the peak usage and offsets
}
