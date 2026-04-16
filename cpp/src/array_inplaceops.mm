#import <Metal/Metal.h>

#include "../include/array_binops.h"
#include "../include/array_inplaceops.h"
#include "../include/metal_source.h"
#include "../include/metal_utils.h"

std::shared_ptr<ArrayHandle> array_inplaceops(const std::shared_ptr<ArrayHandle>& A,
                                              const std::shared_ptr<ArrayHandle>& B,
                                              const std::string& op_name) {
    const auto& shapeA = A->shape();
    const auto& shapeB = B->shape();
    std::vector<int64_t> out_shape = broadcast_shapes(shapeA, shapeB);
    if (out_shape != shapeA) throw std::runtime_error("array_inplaceops: broadcast failed");

    auto defaultForgeHandle = get_default_forge();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)defaultForgeHandle->queue_ptr();

    // compile pipeline on first call
    id<MTLComputePipelineState> pipeline = get_pipeline(op_name, METAL_SOURCE);

    id<MTLBuffer> bufA = (__bridge id<MTLBuffer>)A->metal_buffer();
    id<MTLBuffer> bufB = (__bridge id<MTLBuffer>)B->metal_buffer();

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd)
        throw std::runtime_error(
            "Metal Error: Failed to create command buffer. GPU might out of memory.");
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) throw std::runtime_error("Metal Error: Failed to create command encoder.");
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];

    uint ndim = (uint)shapeA.size();

    if (ndim == 0) {
        uint64_t scalar_shape = 1;
        [enc setBytes:&scalar_shape length:8 atIndex:2];
    } else {
        [enc setBytes:shapeA.data() length:ndim * 8 atIndex:2];
    }
    size_t current_offsetA = A->offset();
    if (ndim == 0) {
        uint64_t scalar_stride = 0;
        [enc setBytes:&scalar_stride length:8 atIndex:3];
    } else {
        [enc setBytes:A->strides().data() length:ndim * 8 atIndex:3];
    }
    [enc setBytes:&current_offsetA length:sizeof(size_t) atIndex:4];
    size_t current_offsetB = B->offset();
    std::vector<int64_t> strides_B = get_bcast_strides(B->shape(), B->strides(), shapeA);
    if (ndim == 0) {
        uint64_t scalar_stride = 0;
        [enc setBytes:&scalar_stride length:8 atIndex:5];
    } else {
        [enc setBytes:strides_B.data() length:ndim * 8 atIndex:5];
    }
    [enc setBytes:&current_offsetB length:sizeof(size_t) atIndex:6];

    if (ndim == 0) ndim = 1;
    [enc setBytes:&ndim length:4 atIndex:7];

    MTLSize grid = MTLSizeMake(numel_from_shape(shapeA), 1, 1);
    MTLSize threads = MTLSizeMake(256, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
    [enc endEncoding];

    [cmd commit];
    A->set_event((__bridge void*)cmd);

    return A;
}
