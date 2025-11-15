#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../include/forge_handle.h"
#include "../include/array_handle.h"
#include "../include/runtime.h"
#include "../include/frontend.h"

namespace py = pybind11;

PYBIND11_MODULE(_backend, m) {
    py::class_<ForgeHandle>(m, "ForgeHandle")
        .def("ir", &ForgeHandle::ir);
    
    m.def("compile_from_source", 
        [](const std::string& src) {
            return compile_from_source_cpp(src);
        },
        py::arg("source")
    );

    m.def("run_kernel", 
        [](ForgeHandle* handle, const std::vector<int>& args) {
            return run_kernel_cpp(handle, args);
        },
        py::arg("kernel"),
        py::arg("args")
    );
}
