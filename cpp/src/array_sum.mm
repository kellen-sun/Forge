#import <Metal/Metal.h>

#include "../include/array_sum.h"
#include "../include/metal_source.h"
#include "../include/metal_utils.h"

std::shared_ptr<ArrayHandle> sum_global(const std::shared_ptr<ArrayHandle>& A, bool keepdims) {
    auto defaultForgeHandle = get_default_forge();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)defaultForgeHandle->queue_ptr();

    id<MTLComputePipelineState> pipeline =
        (__bridge_transfer id<MTLComputePipelineState>)get_pipeline("reduce_sum_global",
                                                                    METAL_SOURCE);

    std::vector<int64_t> out_shape;
    if (keepdims) {
        out_shape = std::vector<int64_t>(A->shape().size(), 1);
    }

    auto out = std::make_shared<ArrayHandle>(out_shape, defaultForgeHandle->device_ptr());

    id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)A->metal_buffer();
    id<MTLBuffer> bufOut = (__bridge id<MTLBuffer>)out->metal_buffer();

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) throw std::runtime_error("Metal Error: Failed to create command buffer.");
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) throw std::runtime_error("Metal Error: Failed to create command encoder.");

    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufOut offset:0 atIndex:1];

    uint in_ndim = (uint)A->shape().size();
    if (in_ndim == 0) {
        uint64_t scalar_shape = 1;
        uint64_t scalar_stride = 0;
        [enc setBytes:&scalar_shape length:8 atIndex:2];
        [enc setBytes:&scalar_stride length:8 atIndex:3];
    } else {
        [enc setBytes:A->shape().data() length:in_ndim * 8 atIndex:2];
        [enc setBytes:A->strides().data() length:in_ndim * 8 atIndex:3];
    }

    size_t current_offsetA = A->offset();
    [enc setBytes:&current_offsetA length:sizeof(size_t) atIndex:4];

    if (in_ndim == 0) in_ndim = 1;
    [enc setBytes:&in_ndim length:4 atIndex:5];

    uint in_numel = numel_from_shape(A->shape());
    [enc setBytes:&in_numel length:4 atIndex:6];

    MTLSize grid = MTLSizeMake(1, 1, 1);
    MTLSize threads = MTLSizeMake(1, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
    [enc endEncoding];

    [cmd commit];
    out->set_event((__bridge void*)cmd);

    return out;
}

std::shared_ptr<ArrayHandle> sum_axis(const std::shared_ptr<ArrayHandle>& A, size_t axis,
                                      bool keepdims) {
    auto defaultForgeHandle = get_default_forge();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)defaultForgeHandle->queue_ptr();

    id<MTLComputePipelineState> pipeline =
        (__bridge_transfer id<MTLComputePipelineState>)get_pipeline("reduce_sum_axis",
                                                                    METAL_SOURCE);

    std::vector<int64_t> out_shape = A->shape();
    if (keepdims) {
        out_shape[axis] = 1;
    } else {
        out_shape.erase(out_shape.begin() + axis);
    }

    auto out = std::make_shared<ArrayHandle>(out_shape, defaultForgeHandle->device_ptr());
    uint out_numel = numel_from_shape(out_shape);

    if (out_numel == 0) return out;

    id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)A->metal_buffer();
    id<MTLBuffer> bufOut = (__bridge id<MTLBuffer>)out->metal_buffer();

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd) throw std::runtime_error("Metal Error: Failed to create command buffer.");
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) throw std::runtime_error("Metal Error: Failed to create command encoder.");

    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufOut offset:0 atIndex:1];

    std::vector<int64_t> kernel_shape = A->shape();
    kernel_shape.erase(kernel_shape.begin() + axis);
    uint kernel_ndim = (uint)kernel_shape.size();

    if (kernel_ndim == 0) {
        uint64_t scalar_shape = 1;
        [enc setBytes:&scalar_shape length:8 atIndex:2];
    } else {
        [enc setBytes:kernel_shape.data() length:kernel_ndim * 8 atIndex:2];
    }
    uint safe_kernel_ndim = kernel_ndim == 0 ? 1 : kernel_ndim;
    [enc setBytes:&safe_kernel_ndim length:4 atIndex:3];

    uint in_ndim = (uint)A->shape().size();
    [enc setBytes:A->shape().data() length:in_ndim * 8 atIndex:4];
    [enc setBytes:A->strides().data() length:in_ndim * 8 atIndex:5];
    size_t current_offsetA = A->offset();
    [enc setBytes:&current_offsetA length:sizeof(size_t) atIndex:6];

    uint u_axis = (uint)axis;
    [enc setBytes:&u_axis length:4 atIndex:7];
    [enc setBytes:&out_numel length:4 atIndex:8];

    MTLSize grid = MTLSizeMake(out_numel, 1, 1);
    MTLSize threads = MTLSizeMake(256, 1, 1);
    if (threads.width > grid.width) {
        threads.width = grid.width;
    }
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
    [enc endEncoding];

    [cmd commit];
    out->set_event((__bridge void*)cmd);

    return out;
}
