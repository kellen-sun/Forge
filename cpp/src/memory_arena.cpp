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
    std::vector<int> last_use(num_nodes, -1);
    for (size_t idx = 0; idx < num_nodes; ++idx) {
        for (auto input : graph.nodes[idx].inputs) {
            last_use[roots[input]] = idx;
        }
    }
    last_use[roots[graph.output_index]] = INT_MAX;
    // 3. Simulate allocation, and frees using greedy best-fit
    // Walk through nodes in graph, if not enough memory available allocate more
    // Recycle dead memory, return the peak usage and offsets
    // If an UPDATE is also its last use, we can edit the data in place, else, need to make a copy
    // for example: x_old = x
    //              x[0] = 5            # Node 2 (UPDATE) -> needs to be a copy
    //              y = x_old + 2
    // Question is: Do we make Copy or Not, whats easier to support
    // If we don't make a copy, we alias track (using the roots), are we guaranteed they're the same
    // node? And points back to same parent? Applies to reshape and transpose as well, we can take a
    // (1,5) arr reshape to (5,1) edit a value, does the original change? (aka: did we make a copy)
    // -> in numpy no COPY seems like the easier model, when a setitem/UPDATE opcode is applied to
    // an object it fully dies (along with all the references to it) and they all become refs to the
    // new obj
}
