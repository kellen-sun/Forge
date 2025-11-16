#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/forge_handle.h"
#include "../include/array_handle.h"
#include "../include/runtime.h"
#include "../include/compiler.h"

namespace py = pybind11;

PYBIND11_MODULE(_backend, m) {
    m.doc() = "Forge";

    py::class_<ForgeHandle>(m, "ForgeHandle")
        .def("ir", &ForgeHandle::ir);

    py::class_<ArrayHandle, std::shared_ptr<ArrayHandle>>(m, "ArrayHandle")
        .def_property_readonly(
            "shape",
            [](const ArrayHandle& a) { return a.shape(); }
        )
        .def_property_readonly(
            "data",
            [](const ArrayHandle& a) { return a.data(); }
        );
    m.def("create_array_from_buffer", &create_array_from_buffer_py, py::arg("buf"), py::arg("shape"));
    m.def("array_shape", &array_shape);
    m.def("array_to_list", &array_to_list);
    
    m.def("compile_from_source", &compile_from_source_cpp);
    m.def("run_kernel", &run_kernel_cpp);
}
