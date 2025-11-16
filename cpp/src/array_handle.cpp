#include "../include/array_handle.h"

std::vector<int64_t> array_shape(const std::shared_ptr<ArrayHandle>& h) {
    return h->shape();
}

std::shared_ptr<ArrayHandle> create_array_from_buffer_py(py::buffer buf, std::vector<int64_t> shape) {
    py::buffer_info info = buf.request();
    if (info.format != py::format_descriptor<float>::format() || info.itemsize != 4) {
        throw std::runtime_error("create_array_from_buffer: buffer must be float32 and contiguous");
    }
    int64_t total = 1;
    for (auto d : shape) total *= d;
    if (info.size != total) {
        throw std::runtime_error("create_array_from_buffer: buffer length doesn't match given shape");
    }
    float* src_ptr = static_cast<float*>(info.ptr);
    std::vector<float> data(src_ptr, src_ptr+info.size);
    return std::make_shared<ArrayHandle>(std::move(data), std::vector<int64_t>(shape.begin(), shape.end()));
}

py::object array_to_list(const std::shared_ptr<ArrayHandle>& h) {
    const auto& shape = h->shape();
    const auto& data = h->data();
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
