#include "../include/memory_arena.h"

MemoryArena::MemoryArena(const Graph& graph, uint64_t element_size) {
    // 1. Calculate array sizes and find roots of each array
    // ---> (root is the original array's memory being used in the case of a view, etc)
    size_t num_nodes = graph.nodes.size();
    // size required for each Array (the node) in bytes
    std::vector<int64_t> sizes(num_nodes);
    this->roots.resize(num_nodes);

    for (size_t idx = 0; idx < num_nodes; ++idx) {
        OpCode op = graph.nodes[idx].op;
        roots[idx] = idx;
        if (op == OpCode::RESHAPE || op == OpCode::TRANSPOSE || op == OpCode::VIEW) {
            int parent = graph.nodes[idx].inputs[0];
            roots[idx] = roots[parent];
        }

        sizes[idx] = element_size * numel_from_shape(graph.nodes[idx].shape);
    }
    int output_root;
    if (num_nodes > 0) output_root = this->roots[graph.output_index];

    // 2. Set last used array, starting at -1, check when last seen inputs
    // Force the output Array's last used to be infinite, so it doesn't get recycled
    std::vector<int> last_use(num_nodes, -1);
    for (size_t idx = 0; idx < num_nodes; ++idx) {
        for (auto input : graph.nodes[idx].inputs) {
            last_use[roots[input]] = idx;
        }
    }
    if (num_nodes > 0) last_use[roots[output_root]] = INT_MAX;

    // 3. Simulate allocation, and frees using greedy best-fit
    // Walk through nodes in graph, if not enough memory available allocate more
    // Recycle dead memory, return the peak usage and offsets
    // Note: we basically never make a copy of the data for transpose,reshape, view, even UPDATE
    // except for reshape when not contiguous i think
    // we just change in place the data and provide a new view
    // so that if we have y = x, then change x, y also changes. y is a reference to it, a view
    // unless we do y = x.copy() or smth, we can provide .copy() support later
    // along with .list(), which would return the python list and allow us
    // to copy by inserting the constant list as a new INPUT in the graph.
    // (so will need support for constants)
    // one issue to beware of later is synchronization of the kernels, if we have y a ref of x
    // then we gotta make sure we dont have readwrite race conditions
    struct FreeBlock {
        uint64_t offset;
        uint64_t size;
    };
    uint64_t peak_memory = 0;
    std::vector<FreeBlock> free_blocks;
    node_offsets.resize(num_nodes, 0);

    for (size_t i = 0; i < num_nodes; ++i) {
        if (roots[i] != i) {
            node_offsets[i] = node_offsets[roots[i]];
            continue;
        }

        if (graph.nodes[i].op == OpCode::INPUT || i == output_root) {
            this->node_offsets[i] = 0;
            continue;
        }

        uint64_t size = sizes[i];
        int best_fit = -1;
        uint64_t min_waste = UINT64_MAX;
        for (int b = 0; b < free_blocks.size(); ++b) {
            if (free_blocks[b].size >= size) {
                uint64_t waste = free_blocks[b].size - size;
                if (waste < min_waste) {
                    min_waste = waste;
                    best_fit = b;
                }
            }
        }
        if (best_fit == -1) {
            node_offsets[i] = peak_memory;
            peak_memory += size;
        } else {
            node_offsets[i] = free_blocks[best_fit].offset;
            free_blocks[best_fit].offset += size;
            free_blocks[best_fit].size -= size;
            if (free_blocks[best_fit].size == 0) free_blocks.erase(free_blocks.begin() + best_fit);
        }

        // Could potentially be optimized by keeping track of a list
        // O(N^2) to O(N)
        for (size_t r = 0; r <= i; ++r) {
            if (roots[r] == r && last_use[r] == i) {
                if (graph.nodes[r].op != OpCode::INPUT && r != output_root) {
                    free_blocks.emplace_back(this->node_offsets[r], sizes[r]);
                }
            }
        }
    }

    this->total_bytes = peak_memory;
}
