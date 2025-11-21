#import <Metal/Metal.h>

#include "../include/array_handle.h"
#include "../include/forge_handle.h"


ArrayHandle::ArrayHandle(std::vector<float> data, std::vector<int64_t> shape, void* dev)
 : data_{std::move(data)}, shape_{std::move(shape)} 
{
    id<MTLDevice> device = (__bridge id<MTLDevice>)dev;

    size_t nbytes = data_.size() * sizeof(float);
    id<MTLBuffer> buf = [device newBufferWithBytes:data_.data()
                                            length:nbytes
                                           options:MTLResourceStorageModeShared];

    metal_buffer_ = (__bridge_retained void*)buf;
}

ArrayHandle::~ArrayHandle() {
    if (metal_buffer_) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)metal_buffer_;
        // ARC releases automatically
        metal_buffer_ = nullptr;
    }
}

static size_t numel_from_shape(const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (auto d : shape) n *= (size_t) d;
    return n;
}

std::vector<float>& ArrayHandle::download() {
    size_t n = numel_from_shape(shape_);
    if (data_.size() != n) throw std::runtime_error("ArrayHandle::download: buffer length doesn't match given shape");
    if (!metal_buffer_) {
        // TODO: Raise a silent warning for not having a metal_buffer_ that can be disabled
        return data_;
    }
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>) metal_buffer_;
    void* src = [buf contents];
    if (src == nullptr) {
        throw std::runtime_error("ArrayHandle::download: buffer contents() null");
    }

    memcpy(data_.data(), src, n * sizeof(float));
    return data_;
}

const std::vector<float>& ArrayHandle::download() const {
    return const_cast<ArrayHandle*>(this)->data();
}

std::vector<int64_t> array_shape(const std::shared_ptr<ArrayHandle>& h) {
    return h->shape();
}

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
    std::vector<float> data(src_ptr, src_ptr+info.size);

    void* dev;
    if (!FH) dev = get_default_forge()->device_ptr();
    else dev = FH->device_ptr();
    return std::make_shared<ArrayHandle>(std::move(data), 
        std::vector<int64_t>(shape.begin(), shape.end()), dev);
}

py::object array_to_list(const std::shared_ptr<ArrayHandle>& h) {
    const auto& shape = h->shape();
    // TODO: Should we always download here?
    const auto& data = h->download();
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
