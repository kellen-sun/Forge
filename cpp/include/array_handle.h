#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/buffer_info.h>

#include <vector>
#include <cstdint>

namespace py = pybind11;

class ArrayHandle {
private:
    std::vector<float> data_;
    std::vector<int64_t> shape_;
public:
    // CONSTRUCTORS //
    ArrayHandle() = default;
    ArrayHandle(std::vector<float>&& data, std::vector<int64_t>&& shape)
     : data_{std::move(data)}, shape_{std::move(shape)} {}
    
    // ACCESSORS //
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<float>& data() const { return data_; }
    std::vector<float>& data() { return data_; }
};

std::vector<int64_t> array_shape(const std::shared_ptr<ArrayHandle>& h);

std::shared_ptr<ArrayHandle> create_array_from_buffer_py(py::buffer buf, std::vector<int64_t> shape);

py::object array_to_list(const std::shared_ptr<ArrayHandle>& h);