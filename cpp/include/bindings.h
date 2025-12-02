#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/buffer_info.h>

#include "array_handle.h"

namespace py = pybind11;


std::shared_ptr<ArrayHandle> create_array_from_buffer_py(py::buffer buf, std::vector<int64_t> shape,
    ForgeHandle* FH);

py::object array_to_list(const ArrayHandle& h);
