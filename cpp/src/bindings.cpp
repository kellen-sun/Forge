#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/forge_handle.h"
#include "../include/array_handle.h"
#include "../include/array_add.h"
#include "../include/runtime.h"
#include "../include/compiler.h"

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
            "data",
            [](const ArrayHandle& a) { return a.data(); }
        );
    m.def("create_array_from_buffer",       [](py::buffer buf, std::vector<int64_t> shape) {
          return create_array_from_buffer_py(buf, shape, /*FH=*/nullptr);
      },
      py::arg("buf"), py::arg("shape"));
    m.def("array_shape", &array_shape);
    m.def("array_to_list", &array_to_list);

    // OPERATIONS //
    m.def("add_arrays", &add_arrays_cpp, py::arg("a"), py::arg("b"));
    
    // COMPILE AND RUN //
    // m.def("compile_from_source", &compile_from_source_cpp);
    // m.def("run_kernel", &run_kernel_cpp);
}
