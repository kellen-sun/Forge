#import <Metal/Metal.h>

#include "../include/array_handle.h"
#include "../include/forge_handle.h"
#include "../include/metal_source.h"
#include "../include/metal_utils.h"

ArrayStorage::~ArrayStorage() {
    if (metal_buffer_) {
        CFRelease(metal_buffer_);
    }
    if (write_event_) {
        CFRelease(write_event_);
    }
}

static std::vector<int64_t> make_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int i = (int)shape.size() - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

ArrayHandle::ArrayHandle(std::vector<int64_t> shape, void* dev)
    : shape_{std::move(shape)}, offset_(0) {
    strides_ = make_strides(shape_);
    size_t nbytes = numel_from_shape(shape_) * sizeof(float);
    storage_ = std::make_shared<ArrayStorage>();
    if (nbytes == 0) {
        return;
    }
    if (!dev) dev = get_default_forge()->device_ptr();
    id<MTLDevice> device = (__bridge id<MTLDevice>)dev;
    id<MTLBuffer> buf = [device newBufferWithLength:nbytes options:MTLResourceStorageModeShared];

    storage_->metal_buffer_ = (__bridge_retained void*)buf;
}

ArrayHandle::ArrayHandle(const float* src_data, std::vector<int64_t> shape, void* dev)
    : shape_{std::move(shape)}, offset_(0) {
    strides_ = make_strides(shape_);
    size_t nbytes = numel_from_shape(shape_) * sizeof(float);
    storage_ = std::make_shared<ArrayStorage>();
    if (nbytes == 0) {
        return;
    }
    if (!dev) dev = get_default_forge()->device_ptr();
    id<MTLDevice> device = (__bridge id<MTLDevice>)dev;
    id<MTLBuffer> buf = [device newBufferWithBytes:src_data
                                            length:nbytes
                                           options:MTLResourceStorageModeShared];

    storage_->metal_buffer_ = (__bridge_retained void*)buf;
}

ArrayHandle::ArrayHandle(const std::shared_ptr<ArrayHandle>& parent, std::vector<int64_t> new_shape,
                         std::vector<int64_t> new_strides, size_t new_offset)
    : shape_(std::move(new_shape)), strides_(std::move(new_strides)), offset_(new_offset) {
    parent->synchronize();
    storage_ = parent->storage_;
}

ArrayHandle::~ArrayHandle() {}

std::span<float> ArrayHandle::data() {
    size_t total = numel_from_shape(shape_);
    if (total == 0) return {};
    if (!storage_->metal_buffer_) {
        throw std::runtime_error(
            "ArrayHandle::data: no existing metal_buffer associated with ArrayHandle");
    }
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)storage_->metal_buffer_;
    float* src = (float*)[buf contents];

    return std::span<float>(src, total);
}

std::span<const float> ArrayHandle::data() const { return const_cast<ArrayHandle*>(this)->data(); }

void ArrayHandle::set_event(void* event) {
    if (storage_->write_event_ == event) return;
    if (storage_->write_event_) CFRelease(storage_->write_event_);
    if (event)
        storage_->write_event_ = (void*)CFRetain(event);
    else
        storage_->write_event_ = nullptr;
}

void* ArrayHandle::metal_buffer() const { return storage_->metal_buffer_; }

void ArrayHandle::set_metal_buffer(void* buf) { storage_->metal_buffer_ = buf; }

void ArrayHandle::copy_from(std::shared_ptr<ArrayHandle> other, std::vector<int64_t> shape,
                            std::vector<int64_t> strides, size_t offset) {
    std::string op_name = "copy_view";
    id<MTLComputePipelineState> pipeline = get_pipeline(op_name, ELEMENTWISE_METAL_SOURCE);

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)get_default_forge()->queue_ptr();
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];

    std::vector<int64_t> src_strides = other->strides();

    if (numel_from_shape(other->shape()) == 1 && numel_from_shape(shape) > 1) {
        src_strides.assign(shape.size(), 0);
    }

    [enc setBuffer:(__bridge id<MTLBuffer>)this->metal_buffer() offset:0 atIndex:0];
    [enc setBuffer:(__bridge id<MTLBuffer>)other->metal_buffer() offset:0 atIndex:1];

    uint ndim = (uint)shape.size();

    [enc setBytes:shape.data() length:ndim * 8 atIndex:2];

    [enc setBytes:strides.data() length:ndim * 8 atIndex:3];
    [enc setBytes:&offset length:8 atIndex:4];

    [enc setBytes:src_strides.data() length:ndim * 8 atIndex:5];
    size_t other_offset = other->offset();
    [enc setBytes:&other_offset length:8 atIndex:6];

    [enc setBytes:&ndim length:4 atIndex:7];

    NSUInteger n_elements = numel_from_shape(shape);
    MTLSize gridSize = MTLSizeMake(n_elements, 1, 1);
    NSUInteger threadGroupSize = MIN(pipeline.maxTotalThreadsPerThreadgroup, n_elements);
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [enc endEncoding];
    [cmd commit];
    this->set_event((__bridge void*)cmd);
}

void ArrayHandle::synchronize() {
    if (!storage_->write_event_) return;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)storage_->write_event_;
    [cmd waitUntilCompleted];
    CFRelease(storage_->write_event_);
    storage_->write_event_ = nullptr;
}

std::vector<int64_t> array_shape(const std::shared_ptr<ArrayHandle>& h) { return h->shape(); }

std::shared_ptr<ArrayHandle> array_reshape(const std::shared_ptr<ArrayHandle>& h,
                                           std::vector<int64_t> shape) {
    bool contiguous = true;
    int64_t z = 1;
    const auto& other_shape = h->shape();
    const auto& other_strides = h->strides();
    for (int i = other_shape.size() - 1; i >= 0; --i) {
        if (other_strides[i] != z) {
            contiguous = false;
            break;
        }
        z *= other_shape[i];
    }
    std::vector<int64_t> new_strides = make_strides(shape);
    if (contiguous) {
        return std::make_shared<ArrayHandle>(h, shape, new_strides, h->offset());
    }
    std::shared_ptr<ArrayHandle> ret = std::make_shared<ArrayHandle>(shape);
    std::vector<int64_t> compact_strides = make_strides(h->shape());
    ret->copy_from(h, other_shape, compact_strides, 0);
    return ret;
}
