#pragma once
#include <cstdint>
#include <vector>

#include "graph.h"

class MemoryArena {
   private:
    uint64_t total_bytes;
    std::vector<uint64_t> node_offsets;

   public:
    // CONSTRUCTORS //
    MemoryArena(const Graph& graph, uint64_t element_size = 4);

    // ACCESSORS //
    const uint64_t get_total_bytes() const { return total_bytes; }
    const std::vector<uint64_t> get_all_offsets() const { return node_offsets; }
    uint64_t get_offset(int node_index) const { return node_offsets[node_index]; }
};
