#include "../include/array_handle.h"
#include "../include/bindings.h"
#include "../include/compiler.h"
#include "../include/graph.h"
#include "../include/memory_arena.h"

namespace nb = nanobind;

std::shared_ptr<ArrayHandle> create_array_from_buffer_py(
    nb::ndarray<float, nb::numpy, nb::c_contig, nb::device::cpu> arr, std::vector<int64_t> shape,
    ForgeHandle* FH) {
    int64_t total = numel_from_shape(shape);
    if (arr.size() != total) {
        throw std::runtime_error(
            "create_array_from_buffer: buffer length doesn't match given shape");
    }
    float* src_ptr = arr.data();
    void* dev = FH ? FH->device_ptr() : get_default_forge()->device_ptr();
    return std::make_shared<ArrayHandle>(src_ptr, shape, dev);
}

nb::object array_to_list(const ArrayHandle& h) {
    const_cast<ArrayHandle&>(h).synchronize();
    const std::vector<int64_t> shape = h.shape();
    const std::vector<int64_t> strides = h.strides();
    const std::span<const float> data = h.data();
    if (shape.empty() || strides.empty()) {
        return nb::cast(data.size() ? data[h.offset()] : 0.0f);
    }

    std::function<nb::object(size_t, size_t)> build;
    build = [&](size_t dim, size_t offset) -> nb::object {
        int64_t stride = strides[dim];
        if (dim + 1 == shape.size()) {
            nb::list lst;
            for (int64_t i = 0; i < shape[dim]; ++i)
                lst.append(nb::float_(data[offset + i * stride]));
            return lst;
        } else {
            nb::list lst;
            for (int64_t i = 0; i < shape[dim]; ++i)
                lst.append(build(dim + 1, offset + i * stride));
            return lst;
        }
    };
    return build(0, h.offset());
}

std::vector<Node> parse_nodes(nb::list flat_nodes) {
    std::vector<Node> nodes;
    nodes.reserve(flat_nodes.size());

    for (auto handle : flat_nodes) {
        auto t = nb::cast<nb::tuple>(handle);

        Node n;
        n.op = static_cast<OpCode>(nb::cast<int>(t[0]));
        n.inputs = nb::cast<std::vector<int>>(t[1]);
        n.shape = nb::cast<std::vector<int64_t>>(t[2]);
        n.offset = nb::cast<int64_t>(t[3]);
        n.strides = nb::cast<std::vector<int64_t>>(t[4]);
        nb::tuple py_args = nb::cast<nb::tuple>(t[5]);

        // different operation add more later, if they take args
        // consider using a switch statement
        if (n.op == OpCode::UPDATE) {
            // py_args = (shape, strides, offset)
            auto s = nb::cast<std::vector<int64_t>>(py_args[0]);
            auto st = nb::cast<std::vector<int64_t>>(py_args[1]);
            int64_t off = nb::cast<int64_t>(py_args[2]);
            // flatten
            n.args.insert(n.args.end(), s.begin(), s.end());
            n.args.insert(n.args.end(), st.begin(), st.end());
            n.args.push_back(off);
        }
        nodes.push_back(n);
    }
    return nodes;
}

std::shared_ptr<Graph> make_graph(nb::list flat_nodes, int output_index) {
    // 1. Get the basic graph
    std::vector<Node> raw_nodes = parse_nodes(flat_nodes);
    // 2. Optimize graph
    std::vector<Node> optimized_nodes = optimize_graph(raw_nodes);
    // 3. Make graph
    // possible that output_index changes after compiling no?
    auto graph = std::make_shared<Graph>(std::move(optimized_nodes), output_index);
    // 4. Get shared memory map (with some Data struct)
    graph->arena = std::make_shared<MemoryArena>(*graph);
    // 5. Compile Graph, to get strings of the relevant kernels and associated info
    generateKernels(*graph);
    // 6. Pre-Compile Metal (MSL -> MTLLibrary)

    return graph;
}
