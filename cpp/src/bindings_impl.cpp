#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/buffer_info.h>

#include "../include/array_handle.h"

namespace py = pybind11;


std::shared_ptr<ArrayHandle> create_array_from_buffer_py(py::buffer buf, std::vector<int64_t> shape,
    ForgeHandle* FH)
{
    py::buffer_info info = buf.request();
    if (info.format != py::format_descriptor<float>::format() || info.itemsize != 4) {
        throw std::runtime_error("create_array_from_buffer: buffer must be float32 and contiguous");
    }
    int64_t total = numel_from_shape(shape);
    if (info.size != total) {
        throw std::runtime_error("create_array_from_buffer: buffer length doesn't match given shape");
    }
    float* src_ptr = static_cast<float*>(info.ptr);
    void* dev = FH ? FH->device_ptr() : get_default_forge()->device_ptr();
    std::shared_ptr<ArrayHandle> handle = std::make_shared<ArrayHandle>(src_ptr, shape, dev);
    return handle;
}

py::object array_to_list(const ArrayHandle& h) {
    const_cast<ArrayHandle&>(h).synchronize();
    const std::vector<int64_t> shape = h.shape();
    const std::span<const float> data = h.data();
    if (shape.empty()) {
        return py::float_(data.size() ? data[0] : 0.0);
    }

    std::function<py::object(size_t, size_t)> build = [&](size_t dim, size_t offset) -> py::object {
        if (dim + 1 == shape.size()) {
            py::list lst;
            for (int64_t i = 0; i < shape[dim]; ++i) 
                lst.append(py::float_(data[offset + i]));
            return lst;
        } else {
            py::list lst;
            int64_t stride = 1;
            for (size_t k = dim + 1; k < shape.size(); ++k) stride *= shape[k];
            for (int64_t i = 0; i < shape[dim]; ++i)
                lst.append(build(dim + 1, offset + i * stride));
            return lst;
        }
    };
    return build(0, 0);
}
