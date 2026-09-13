#include <vector>

#include "../include/pass.h"

void DCEPass::run(IR& ir) {
    const int n = static_cast<int>(ir.nodes.size());
    if (n == 0) return;
    std::vector<char> live(n, 0);

    std::vector<int> stack = {ir.output_index};
    while (!stack.empty()) {
        int i = stack.back();
        stack.pop_back();
        if (i < 0 || i >= n || live[i]) continue;
        live[i] = 1;
        for (int inp : ir.nodes[i].inputs) stack.push_back(inp);
    }

    // execute() indexes Python args by INPUT node id — keep unused INPUTs
    for (int i = 0; i < n; ++i) {
        if (ir.nodes[i].op == OpCode::INPUT) live[i] = 1;
    }

    std::vector<int> remap(n, -1);
    std::vector<Node> kept;
    kept.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (!live[i]) continue;
        remap[i] = static_cast<int>(kept.size());
        kept.push_back(ir.nodes[i]);
    }
    for (auto& node : kept) {
        for (int& inp : node.inputs) inp = remap[inp];
    }
    ir.output_index = remap[ir.output_index];
    ir.nodes = std::move(kept);
}
