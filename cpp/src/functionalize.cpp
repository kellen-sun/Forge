#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../include/ir_utils.h"
#include "../include/pass.h"

static bool same_layout(const Node& a, const Node& b) {
    return a.shape == b.shape && a.strides == b.strides && a.offset == b.offset;
}

void FunctionalizePass::run(IR& ir) {
    std::vector<Node> out;
    std::vector<Node> traced;
    std::vector<int> remap(ir.nodes.size());
    std::unordered_map<int, int> latest;
    std::map<std::pair<int, int>, int> replayed;

    auto emit = [&](Node node, const Node& layout) {
        traced.push_back(layout);
        out.push_back(std::move(node));
        return static_cast<int>(out.size()) - 1;
    };

    auto current = [&](int value) {
        auto it = latest.find(storage_root(out, value));
        if (it == latest.end()) return value;
        const int version = it->second;
        if (layout_version(out, value) == version && same_layout(out[value], traced[value])) {
            return value;
        }
        if (same_layout(out[version], traced[value])) return version;

        auto [slot, inserted] = replayed.emplace(std::make_pair(version, value), -1);
        if (inserted) {
            Node view;
            view.op = OpCode::VIEW;
            view.inputs = {version};
            view.shape = traced[value].shape;
            view.strides = traced[value].strides;
            view.offset = traced[value].offset;
            slot->second = emit(view, view);
        }
        return slot->second;
    };

    for (int i = 0; i < static_cast<int>(ir.nodes.size()); ++i) {
        Node node = ir.nodes[i];
        const Node layout = node;
        for (int& input : node.inputs) input = current(remap[input]);

        if (node.op == OpCode::UPDATE) {
            const int version = layout_version(out, node.inputs[0]);
            node.inputs[0] = version;
            node.shape = out[version].shape;
            node.strides = out[version].strides;
            node.offset = out[version].offset;
            remap[i] = emit(std::move(node), layout);
            latest[storage_root(out, remap[i])] = remap[i];
        } else {
            remap[i] = emit(std::move(node), layout);
        }
    }

    ir.output_index = current(remap[ir.output_index]);
    ir.nodes = std::move(out);
}
