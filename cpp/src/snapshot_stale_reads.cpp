#include <algorithm>
#include <unordered_map>
#include <vector>

#include "../include/array_handle.h"
#include "../include/ir_utils.h"
#include "../include/pass.h"

struct SnapshotStaleReads {
    const std::vector<Node>& nodes;
    std::unordered_map<int, int> overwritten_at;
    std::unordered_map<int, std::vector<int>> snapshots;
    std::vector<Node> out;
    std::vector<int> remap;
    std::unordered_map<int, int> snapshot_of;

    int stale_since(int value, int reader) const {
        auto it = overwritten_at.find(layout_version(nodes, value));
        if (it == overwritten_at.end()) return -1;
        if (it->second >= reader) return -1;
        return it->second;
    }

    void collect(int value, int reader) {
        const int update = stale_since(value, reader);
        if (update < 0) return;
        std::vector<int>& values = snapshots[update];
        if (std::find(values.begin(), values.end(), value) == values.end()) {
            values.push_back(value);
        }
    }

    int resolve(int value, int reader) const {
        if (stale_since(value, reader) >= 0) {
            return snapshot_of.at(value);
        }
        return remap[value];
    }
};

void SnapshotStaleReadsPass::run(IR& ir) {
    const int n = static_cast<int>(ir.nodes.size());
    SnapshotStaleReads state{ir.nodes, {}, {}, {}, std::vector<int>(n, -1), {}};

    for (int i = 0; i < n; ++i) {
        if (ir.nodes[i].op == OpCode::UPDATE) {
            state.overwritten_at.emplace(layout_version(ir.nodes, ir.nodes[i].inputs[0]), i);
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int input : ir.nodes[i].inputs) state.collect(input, i);
    }
    state.collect(ir.output_index, n);
    if (state.snapshots.empty()) return;

    for (int i = 0; i < n; ++i) {
        auto it = state.snapshots.find(i);
        if (it != state.snapshots.end()) {
            for (int value : it->second) {
                Node copy;
                copy.op = OpCode::COPY;
                copy.inputs = {state.remap[value]};
                copy.shape = ir.nodes[value].shape;
                copy.strides = make_strides(copy.shape);
                copy.offset = 0;
                state.snapshot_of[value] = static_cast<int>(state.out.size());
                state.out.push_back(std::move(copy));
            }
        }
        Node node = ir.nodes[i];
        for (int& input : node.inputs) input = state.resolve(input, i);
        state.remap[i] = static_cast<int>(state.out.size());
        state.out.push_back(std::move(node));
    }

    ir.output_index = state.resolve(ir.output_index, n);
    ir.nodes = std::move(state.out);
}
