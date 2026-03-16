#include <sstream>
#include <numeric>
#include "../include/compiler.h"
#include "../include/common.h"

std::vector<Node> optimize_graph(std::vector<Node> raw_nodes) { return raw_nodes; }
// Could generate Fused Kernels, with special OpCodes
// Description of which fusedkernel for the OpCode given in the OpCodes "Arg" parameter
// Read about MLIR & TVM as options here instead of doing it here
// Optimization ideas:
// 1. dead code elimination
// 2. fold constants, (3 + 4) known at compile time. or
// 3. common sub expression elimination
// ---> those all fall under LVN analysis
// 4. reroll a loop, say they did: for i in len(A): A[i] + B[i] -> just A+ B yk
// 5. Fusion, combine nodes into "blocks" that run in the same "way" (elementwise easiest)
// 6. loop fusion. like two for i in range(100) can be put together



// Generates one huge string of all the kernel functions back to back
// if the op requires no gpu kernel (INPUT, VIEW, etc -> call it "no op"),
// we dont generate any string
// for each node in the graph, we save the associated kernel's name, in the config.name
// if it's a no-op, keep a "ghost" empty config
// so that at the end: configs.size == nodes.size (== pipelines.size)
// Op metadata: Shape, strides (internal offset?) WONT be handled from execute()
// This means the kernel strings we generate needs to bake in/hardcode the loops for that
// kernels generated so that out buffer is idx 0, then the N inputs to the node (in order)


void generateKernels(Graph& graph) {
    graph.shader_source = "#include <metal_stdlib>\nusing namespace metal;\n";
    for (int64_t i = 0; i < graph.nodes.size(); i++) {
        Node &node = graph.nodes[i];

        if (node.op == OpCode::INPUT || node.op == OpCode::VIEW || node.op == OpCode::RESHAPE || node.op == OpCode::TRANSPOSE) {
            // if Input, do a no-op
            graph.configs.push_back({});
            continue;
        }
        
        // function definition and inputs
        std::ostringstream oss;
        std::string kernel_name = "op_" + std::to_string(i);
        oss << "kernel void " << kernel_name << " (\n\
                device float* out [[ buffer(0) ]],\n";
        for (int64_t j = 0; j < node.inputs.size(); j++) {
            oss << "const device float* in" << j 
                << " [[ buffer(" + std::to_string(j + 1) + ") ]],\n";
        }
        oss << "uint gid [[ thread_position_in_grid ]])\n{\n";

        // hardcode constants: shape array
        oss << "constant long shape[] = {";
        for (int64_t j = 0; j < node.shape.size(); j++) {
            oss << node.shape[j] << ",}"[j == node.shape.size() - 1];
        }
        oss << ";\n";

        // hardcode constants: strides arrays
        for (int64_t j = 0; j < node.inputs.size(); j++) {
            oss << "constant long strides_in" << j << "[] = {";
            Node &input = graph.nodes[node.inputs[j]];
            for (int64_t k = 0; k < input.strides.size(); k++) {
                oss << input.strides[k] << ",}"[k == input.strides.size() - 1];
            }
            oss << ";\n";
        }

        // declare variables for computing strided read indices
        oss << "uint remaining = gid;\n";
        for (int64_t j = 0; j < node.inputs.size(); j++) {
            oss << "uint idx_in" << j << " = 0;\n"; // check that this is actually 0, is it actually zero??? Should be, I think 
        }

        // compute strided read indices
        oss << "for (int i = " << node.shape.size() - 1 << "; i >= 0; i--) {\n";
        oss << "uint coord = remaining \% shape[i];\n";
        for (int64_t j = 0; j < node.inputs.size(); j++) {
            oss << "idx_in" << j << " += coord * strides_in" << j << "[i];\n";
        }
        oss << "remaining /= shape[i];\n}\n";

        // note that I'm just putting the 4 binary ops for now. Need to do the others too but that's later
        oss << "out[gid] = ";
        oss << "in0[idx_in0] " << op_symbol.at(node.op) << " in1[idx_in1];\n}\n\n";
        
        graph.shader_source += oss.str();
        uint64_t total_elements = numel_from_shape(node.shape);
        graph.configs.push_back({.name = kernel_name, .grid = {total_elements, 1, 1}, .group = {256, 1, 1}});
    }
}
