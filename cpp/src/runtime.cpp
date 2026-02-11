#include "../include/graph.h"
#include "../include/memory_arena.h"

std::shared_ptr<ArrayHandle> Graph::execute(std::vector<std::shared_ptr<ArrayHandle>> inputs) {
    // Get shared memory map (with some Data struct) -> runtime.cpp
    MemoryArena arena(*this);
    // Compile Graph, to get strings of the relevant kernels and associated info (new DS again)
    // -> write this in compiler.cpp
    // Combine this information with inputs to launch -> use this function
}
