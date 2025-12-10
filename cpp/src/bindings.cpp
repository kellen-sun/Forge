#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/forge_handle.h"
#include "../include/array_handle.h"
#include "../include/array_binops.h"
#include "../include/runtime.h"
#include "../include/compiler.h"
#include "../include/bindings.h"

namespace py = pybind11;

PYBIND11_MODULE(_backend, m) {
    // DOC //
    m.doc() = "Forge";

    // FORGE HANDLE //
    py::class_<ForgeHandle>(m, "ForgeHandle")
        .def("ir", &ForgeHandle::ir);

    // ARRAY HANDLE //
    py::class_<ArrayHandle, std::shared_ptr<ArrayHandle>>(m, "ArrayHandle")
        .def_property_readonly(
            "shape",
            [](const ArrayHandle& a) { return a.shape(); }
        )
        .def_property_readonly(
            "strides",
            [](const ArrayHandle& a) { return a.strides(); }
        )
        .def_property_readonly(
            "offset",
            [](const ArrayHandle& a) { return a.offset(); }
        )
        .def_property_readonly(
            "data",
            [](const ArrayHandle& a) { return a.data(); }
        )
        .def("item", [](ArrayHandle& h) -> float {
        if (!h.shape().empty()) {
            throw std::runtime_error("item(): can only convert scalar arrays to float");
        }
        h.synchronize();

        return h.data()[h.offset()];
    });
    m.def("create_array_from_buffer", [](py::buffer buf, std::vector<int64_t> shape) {
          return create_array_from_buffer_py(buf, shape, /*FH=*/nullptr);
      },
      py::arg("buf"), py::arg("shape"));
    m.def("make_view", [](std::shared_ptr<ArrayHandle> h, std::vector<int64_t> shape, 
                        std::vector<int64_t> strides, size_t offset) {
        return std::make_shared<ArrayHandle>(h, shape, strides, offset);
    });
    m.def("array_shape", &array_shape);
    m.def("array_to_list", &array_to_list);

    // OPERATIONS //
    m.def("add", [](const std::shared_ptr<ArrayHandle>& a, const std::shared_ptr<ArrayHandle>& b)
     { return binops_arrays_cpp(a, b, "add"); });
    m.def("sub", [](const std::shared_ptr<ArrayHandle>& a, const std::shared_ptr<ArrayHandle>& b)
     { return binops_arrays_cpp(a, b, "sub"); });
    m.def("mul", [](const std::shared_ptr<ArrayHandle>& a, const std::shared_ptr<ArrayHandle>& b)
     { return binops_arrays_cpp(a, b, "mul"); });
    m.def("div", [](const std::shared_ptr<ArrayHandle>& a, const std::shared_ptr<ArrayHandle>& b)
     { return binops_arrays_cpp(a, b, "div"); });
    
    // COMPILE AND RUN //
    // m.def("compile_from_source", &compile_from_source_cpp);
    // m.def("run_kernel", &run_kernel_cpp);
}
