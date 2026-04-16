#import <Metal/Metal.h>

#include "../include/array_unaryops.h"
#include "../include/metal_source.h"
#include "../include/metal_utils.h"

std::shared_ptr<ArrayHandle> array_unaryops(const std::shared_ptr<ArrayHandle>& A,
                                            const std::string& op_name) {
    auto defaultForgeHandle = get_default_forge();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)defaultForgeHandle->queue_ptr();

    // compile pipeline on first call
    id<MTLComputePipelineState> pipeline = get_pipeline(op_name, METAL_SOURCE);

    // allocate output ArrayHandle
    auto out = std::make_shared<ArrayHandle>(A->shape(), defaultForgeHandle->device_ptr());

    id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)A->metal_buffer();
    id<MTLBuffer> bufOut = (__bridge id<MTLBuffer>)out->metal_buffer();

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd)
        throw std::runtime_error(
            "Metal Error: Failed to create command buffer. GPU might out of memory.");
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) throw std::runtime_error("Metal Error: Failed to create command encoder.");
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufOut offset:0 atIndex:1];

    uint ndim = (uint)out->shape().size();

    if (ndim == 0) {
        uint64_t scalar_shape = 1;
        [enc setBytes:&scalar_shape length:8 atIndex:2];
    } else {
        [enc setBytes:out->shape().data() length:ndim * 8 atIndex:2];
    }
    size_t current_offsetA = A->offset();
    if (ndim == 0) {
        uint64_t scalar_stride = 0;
        [enc setBytes:&scalar_stride length:8 atIndex:3];
    } else {
        [enc setBytes:A->strides().data() length:ndim * 8 atIndex:3];
    }
    [enc setBytes:&current_offsetA length:sizeof(size_t) atIndex:4];

    if (ndim == 0) ndim = 1;
    [enc setBytes:&ndim length:4 atIndex:5];

    MTLSize grid = MTLSizeMake(numel_from_shape(A->shape()), 1, 1);
    MTLSize threads = MTLSizeMake(256, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
    [enc endEncoding];

    [cmd commit];
    out->set_event((__bridge void*)cmd);

    return out;
}
