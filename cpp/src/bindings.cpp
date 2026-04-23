#include "../include/bindings.h"

#include "../include/array_binops.h"
#include "../include/array_handle.h"
#include "../include/compiler.h"
#include "../include/graph.h"

namespace nb = nanobind;

NB_MODULE(_backend, m) {
    // DOC //
    m.doc() = "Forge";

    // ARRAY HANDLE //
    nb::class_<ArrayHandle>(m, "ArrayHandle")
        .def_prop_ro("shape", [](const ArrayHandle& h) { return h.shape(); })
        .def_prop_ro("strides", [](const ArrayHandle& h) { return h.strides(); })
        .def_prop_ro("offset", [](const ArrayHandle& h) { return h.offset(); })
        .def_prop_ro("data", [](const ArrayHandle& h) { return h.data(); })
        .def("item", [](ArrayHandle& h) -> float {
            if (!h.shape().empty()) {
                throw std::runtime_error("item(): can only convert scalar arrays to float");
            }
            h.synchronize();
            return h.data()[h.offset()];
        });
    m.def(
        "create_array_from_buffer",
        [](nb::ndarray<float, nb::numpy, nb::c_contig, nb::device::cpu> arr,
           std::vector<int64_t> shape) {
            return create_array_from_buffer_py(arr, shape, /*FH=*/nullptr);
        },
        nb::arg("arr"), nb::arg("shape"));
    m.def("make_view", [](std::shared_ptr<ArrayHandle> h, std::vector<int64_t> shape,
                          std::vector<int64_t> strides, size_t offset) {
        return std::make_shared<ArrayHandle>(h, shape, strides, offset);
    });
    m.def("copy_to_view", [](std::shared_ptr<ArrayHandle> h, std::shared_ptr<ArrayHandle> other,
                             std::vector<int64_t> shape, std::vector<int64_t> strides,
                             size_t offset) { h->copy_from(other, shape, strides, offset); });
    m.def("reshape", &array_reshape);
    m.def("array_shape", &array_shape);
    m.def("array_to_list", &array_to_list);
    m.def("infer_shape_and_flatten", &infer_shape_and_flatten_py, nb::arg("data"));

    // OPERATIONS //
    m.def("add", [](const std::shared_ptr<ArrayHandle>& a, const std::shared_ptr<ArrayHandle>& b) {
        return array_binops(a, b, "add");
    });
    m.def("sub", [](const std::shared_ptr<ArrayHandle>& a, const std::shared_ptr<ArrayHandle>& b) {
        return array_binops(a, b, "sub");
    });
    m.def("mul", [](const std::shared_ptr<ArrayHandle>& a, const std::shared_ptr<ArrayHandle>& b) {
        return array_binops(a, b, "mul");
    });
    m.def("div", [](const std::shared_ptr<ArrayHandle>& a, const std::shared_ptr<ArrayHandle>& b) {
        return array_binops(a, b, "div");
    });
    m.def("matmul", [](const std::shared_ptr<ArrayHandle>& a,
                       const std::shared_ptr<ArrayHandle>& b) { return array_matmul(a, b); });

    // COMPILE AND RUN //
    nb::class_<Graph>(m, "Graph").def("execute", &Graph::execute);
    m.def("make_graph", &make_graph);
}
