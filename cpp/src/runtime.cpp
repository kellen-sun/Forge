#include "../include/compiler.h"
#include "../include/graph.h"
#include "../include/memory_arena.h"

std::shared_ptr<ArrayHandle> Graph::execute(std::vector<std::shared_ptr<ArrayHandle>> inputs) {
    // Combine this information with inputs to execute
    // a) Allocate the memory plan needed
    // b) Copy the input data over
    // c) Go through the generated kernel strings, compile them if not cached
    // d) Set the correct buffer input offsets and strides, etc
    // e) Launch the kernels one by one - Noting when synchs are necessary
    // ---- Like do we "need" to synch if next kernel on same memory? (or on diff mem)
    // f) Copy output Array out, to free the huge MemoryArena
}
