#import <Metal/Metal.h>

#include "../include/array_handle.h"
#include "../include/forge_handle.h"


ArrayHandle::ArrayHandle(std::vector<int64_t> shape, void* dev)
    : shape_{std::move(shape)} 
{
    size_t nbytes = numel_from_shape(shape_) * sizeof(float);
    if (nbytes == 0) {
        metal_buffer_ = nullptr;
        return;
    }
    if (!dev) dev = get_default_forge()->device_ptr();
    id<MTLDevice> device = (__bridge id<MTLDevice>)dev;
    id<MTLBuffer> buf = [device newBufferWithLength:nbytes
                                           options:MTLResourceStorageModeShared];

    metal_buffer_ = (__bridge_retained void*)buf;
}

ArrayHandle::ArrayHandle(const float* src_data, std::vector<int64_t> shape, void* dev)
    : shape_{std::move(shape)} 
{
    size_t nbytes = numel_from_shape(shape_) * sizeof(float);
    if (nbytes == 0) {
        metal_buffer_ = nullptr;
        return;
    }
    if (!dev) dev = get_default_forge()->device_ptr();
    id<MTLDevice> device = (__bridge id<MTLDevice>)dev;
    id<MTLBuffer> buf = [device newBufferWithBytes:src_data
                                            length:nbytes
                                           options:MTLResourceStorageModeShared];

    metal_buffer_ = (__bridge_retained void*)buf;
}

ArrayHandle::~ArrayHandle() {
    if (metal_buffer_) {
        CFRelease(metal_buffer_);
        metal_buffer_ = nullptr;
    }
}

std::span<float> ArrayHandle::data() {
    size_t total = numel_from_shape(shape_);
    if (total == 0) return {};
    if (!metal_buffer_) {
        throw std::runtime_error("ArrayHandle::data: no existing metal_buffer associated with ArrayHandle");
    }
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>) metal_buffer_;
    float* src = (float*)[buf contents];

    return std::span<float>(src, total);
}

std::span<const float> ArrayHandle::data() const {
    return const_cast<ArrayHandle*>(this)->data();
}

std::vector<int64_t> array_shape(const std::shared_ptr<ArrayHandle>& h) {
    return h->shape();
}
