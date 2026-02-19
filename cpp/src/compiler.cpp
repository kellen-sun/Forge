#include "../include/compiler.h"

std::vector<Node> optimize_graph(std::vector<Node> raw_nodes) { return raw_nodes; }
// Could generate Fused Kernels, with special OpCodes
// Description of which fusedkernel for the OpCode given in the OpCodes "Arg" parameter

void generateKernels(Graph& graph) {}
// Generates one huge string of all the kernel functions back to back
// if the op requires no gpu kernel (INPUT, VIEW, etc -> call it "no op"),
// we dont generate any string
// for each node in the graph, we save the associated kernel's name, in the config.name
// if it's a no-op, keep a "ghost" empty config
// so that at the end: configs.size == nodes.size (== pipelines.size)
// Op metadata: Shape, strides (internal offset?) WONT be handled from execute()
// This means the kernel strings we generate needs to bake in/hardcode the loops for that
// kernels generated so that out buffer is idx 0, then the N inputs to the node (in order)
