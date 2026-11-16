#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../include/ir_utils.h"
#include "../include/pass.h"

struct Functionalize {
    std::vector<Node> out;
    std::vector<Node> traced;
    std::unordered_map<int, int> latest;
    std::map<std::pair<int, int>, int> replayed;

    int emit(Node node, const Node& layout) {
        traced.push_back(layout);
        out.push_back(std::move(node));
        return static_cast<int>(out.size()) - 1;
    }

    int current(int value) {
        auto it = latest.find(storage_root(out, value));
        if (it == latest.end()) return value;
        const int version = it->second;
        const bool same_version = (layout_version(out, value) == version);
        if (same_version && same_layout(out[value], traced[value])) return value;
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
    }
};

void FunctionalizePass::run(IR& ir) {
    Functionalize state;
    std::vector<int> remap(ir.nodes.size());

    for (int i = 0; i < static_cast<int>(ir.nodes.size()); ++i) {
        Node node = ir.nodes[i];
        const Node layout = node;
        for (int& input : node.inputs) input = state.current(remap[input]);

        if (node.op == OpCode::UPDATE) {
            const int version = layout_version(state.out, node.inputs[0]);
            node.inputs[0] = version;
            node.shape = state.out[version].shape;
            node.strides = state.out[version].strides;
            node.offset = state.out[version].offset;
            remap[i] = state.emit(std::move(node), layout);
            state.latest[storage_root(state.out, remap[i])] = remap[i];
        } else {
            remap[i] = state.emit(std::move(node), layout);
        }
    }

    ir.output_index = state.current(remap[ir.output_index]);
    ir.nodes = std::move(state.out);
}
