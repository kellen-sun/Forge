#include <pybind11/buffer_info.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/array_handle.h"
#include "../include/compiler.h"
#include "../include/graph.h"

namespace py = pybind11;

std::shared_ptr<ArrayHandle> create_array_from_buffer_py(py::buffer buf, std::vector<int64_t> shape,
                                                         ForgeHandle* FH) {
    py::buffer_info info = buf.request();
    if (info.format != py::format_descriptor<float>::format() || info.itemsize != 4) {
        throw std::runtime_error("create_array_from_buffer: buffer must be float32 and contiguous");
    }
    int64_t total = numel_from_shape(shape);
    if (info.size != total) {
        throw std::runtime_error(
            "create_array_from_buffer: buffer length doesn't match given shape");
    }
    float* src_ptr = static_cast<float*>(info.ptr);
    void* dev = FH ? FH->device_ptr() : get_default_forge()->device_ptr();
    std::shared_ptr<ArrayHandle> handle = std::make_shared<ArrayHandle>(src_ptr, shape, dev);
    return handle;
}

py::object array_to_list(const ArrayHandle& h) {
    const_cast<ArrayHandle&>(h).synchronize();
    const std::vector<int64_t> shape = h.shape();
    const std::vector<int64_t> strides = h.strides();
    const std::span<const float> data = h.data();
    if (shape.empty() || strides.empty()) {
        return py::float_(data.size() ? data[h.offset()] : 0.0);
    }

    std::function<py::object(size_t, size_t)> build;
    build = [&](size_t dim, size_t offset) -> py::object {
        int64_t stride = strides[dim];
        if (dim + 1 == shape.size()) {
            py::list lst;
            for (int64_t i = 0; i < shape[dim]; ++i)
                lst.append(py::float_(data[offset + i * stride]));
            return lst;
        } else {
            py::list lst;
            for (int64_t i = 0; i < shape[dim]; ++i)
                lst.append(build(dim + 1, offset + i * stride));
            return lst;
        }
    };
    return build(0, h.offset());
}

std::shared_ptr<Graph> make_graph_wrapper(py::list flat_nodes, int output_index) {
    std::vector<Node> nodes;
    nodes.reserve(flat_nodes.size());

    for (auto handle : flat_nodes) {
        auto t = handle.cast<py::tuple>();

        Node n;
        n.op = static_cast<OpCode>(t[0].cast<int>());
        n.inputs = t[1].cast<std::vector<int>>();
        n.shape = t[2].cast<std::vector<uint64_t>>();
        n.offset = t[3].cast<uint64_t>();
        n.strides = t[4].cast<std::vector<uint64_t>>();

        py::tuple py_args = t[5].cast<py::tuple>();
        // different operation add more later, if they take args
        // consider using a switch statement
        if (n.op == OpCode::UPDATE) {
            // py_args = (shape, strides, offset)
            auto s = py_args[0].cast<std::vector<uint64_t>>();
            auto st = py_args[1].cast<std::vector<uint64_t>>();
            uint64_t off = py_args[2].cast<uint64_t>();
            // flatten
            n.args.insert(n.args.end(), s.begin(), s.end());
            n.args.insert(n.args.end(), st.begin(), st.end());
            n.args.push_back(off);
        }
        nodes.push_back(n);
    }

    std::vector<Node> optimized_nodes = optimize_graph(nodes);
    // possible that output_index changes after compiling no?
    return std::make_shared<Graph>(std::move(optimized_nodes), output_index);
}
