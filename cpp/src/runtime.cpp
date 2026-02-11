#include "../include/graph.h"
#include "../include/memory_arena.h"

std::shared_ptr<ArrayHandle> Graph::execute(std::vector<std::shared_ptr<ArrayHandle>> inputs) {
    // 1. Get shared memory map (with some Data struct) -> runtime.cpp
    MemoryArena arena(*this);
    // 2. Compile Graph, to get strings of the relevant kernels and associated info (new DS again)
    // -> compiler.cpp
    // 3. Combine this information with inputs to launch -> launch
    // a) Allocate the memory plan needed
    // b) Copy the input data over
    // c) Go through the generated kernel strings, compile them if not cached
    // d) Set the correct buffer input offsets and strides, etc
    // e) Launch the kernels one by one - Noting when synchs are necessary
    // ---- Like do we "need" to synch if next kernel on same memory? (or on diff mem)
    // f) Copy output Array out, to free the huge MemoryArena
}
