#include <algorithm>
#include <unordered_map>
#include <vector>

#include "../include/array_handle.h"
#include "../include/ir_utils.h"
#include "../include/pass.h"

void SnapshotStaleReadsPass::run(IR& ir) {
    const int n = static_cast<int>(ir.nodes.size());

    std::unordered_map<int, int> overwritten_at;
    for (int i = 0; i < n; ++i) {
        if (ir.nodes[i].op == OpCode::UPDATE) {
            overwritten_at.emplace(layout_version(ir.nodes, ir.nodes[i].inputs[0]), i);
        }
    }

    auto stale_since = [&](int value, int reader) {
        auto it = overwritten_at.find(layout_version(ir.nodes, value));
        return (it != overwritten_at.end() && it->second < reader) ? it->second : -1;
    };

    std::unordered_map<int, std::vector<int>> snapshots;
    auto collect = [&](int value, int reader) {
        const int update = stale_since(value, reader);
        if (update < 0) return;
        auto& values = snapshots[update];
        if (std::find(values.begin(), values.end(), value) == values.end()) {
            values.push_back(value);
        }
    };
    for (int i = 0; i < n; ++i) {
        for (int input : ir.nodes[i].inputs) collect(input, i);
    }
    collect(ir.output_index, n);
    if (snapshots.empty()) return;

    std::vector<Node> out;
    std::vector<int> remap(n, -1);
    std::unordered_map<int, int> snapshot_of;
    auto resolve = [&](int value, int reader) {
        return stale_since(value, reader) >= 0 ? snapshot_of.at(value) : remap[value];
    };

    for (int i = 0; i < n; ++i) {
        auto it = snapshots.find(i);
        if (it != snapshots.end()) {
            for (int value : it->second) {
                Node copy;
                copy.op = OpCode::COPY;
                copy.inputs = {remap[value]};
                copy.shape = ir.nodes[value].shape;
                copy.strides = make_strides(copy.shape);
                copy.offset = 0;
                snapshot_of[value] = static_cast<int>(out.size());
                out.push_back(std::move(copy));
            }
        }
        Node node = ir.nodes[i];
        for (int& input : node.inputs) input = resolve(input, i);
        remap[i] = static_cast<int>(out.size());
        out.push_back(std::move(node));
    }

    ir.output_index = resolve(ir.output_index, n);
    ir.nodes = std::move(out);
}
