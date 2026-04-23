#import <Metal/Metal.h>

#include "../include/array_elementwise.h"
#include "../include/broadcast.h"
#include "../include/metal_source.h"
#include "../include/metal_utils.h"

std::shared_ptr<ArrayHandle> array_unaryops(const std::shared_ptr<ArrayHandle>& A,
                                            const std::string& op_name) {
    auto fh = get_default_forge();
    auto out = std::make_shared<ArrayHandle>(A->shape(), fh->device_ptr());

    id<MTLCommandBuffer> cmd = launch_elementwise(
        op_name, A->shape(),
        {{(__bridge id<MTLBuffer>)A->metal_buffer(), A->shape(), A->strides(), A->offset()}},
        (__bridge id<MTLBuffer>)out->metal_buffer());
    out->set_event((__bridge void*)cmd);
    return out;
}

std::shared_ptr<ArrayHandle> array_binops(const std::shared_ptr<ArrayHandle>& A,
                                          const std::shared_ptr<ArrayHandle>& B,
                                          const std::string& op_name) {
    const auto& shapeA = A->shape();
    const auto& shapeB = B->shape();
    std::vector<int64_t> out_shape =
        (shapeA == shapeB) ? shapeA : broadcast_shapes(shapeA, shapeB);

    auto fh = get_default_forge();
    auto out = std::make_shared<ArrayHandle>(out_shape, fh->device_ptr());

    id<MTLCommandBuffer> cmd = launch_elementwise(
        op_name, out_shape,
        {
            {(__bridge id<MTLBuffer>)A->metal_buffer(), shapeA, A->strides(), A->offset()},
            {(__bridge id<MTLBuffer>)B->metal_buffer(), shapeB, B->strides(), B->offset()},
        },
        (__bridge id<MTLBuffer>)out->metal_buffer());
    out->set_event((__bridge void*)cmd);
    return out;
}

std::shared_ptr<ArrayHandle> array_inplaceops(const std::shared_ptr<ArrayHandle>& A,
                                              const std::shared_ptr<ArrayHandle>& B,
                                              const std::string& op_name) {
    const auto& shapeA = A->shape();
    const auto& shapeB = B->shape();
    std::vector<int64_t> out_shape = broadcast_shapes(shapeA, shapeB);
    if (out_shape != shapeA) throw std::runtime_error("array_inplaceops: broadcast failed");

    id<MTLCommandBuffer> cmd = launch_elementwise(
        op_name, shapeA,
        {
            {(__bridge id<MTLBuffer>)A->metal_buffer(), shapeA, A->strides(), A->offset()},
            {(__bridge id<MTLBuffer>)B->metal_buffer(), shapeB, B->strides(), B->offset()},
        },
        nil);
    A->set_event((__bridge void*)cmd);
    return A;
}

// Nullary ops (rand/randn/zeros) don't share the strided-input buffer layout
// used by the other elementwise ops: their kernels index directly by gid and
// only take (Out, seed). Keep them as a dedicated launcher rather than forcing
// them through launch_elementwise.
std::shared_ptr<ArrayHandle> array_nullaryops(const std::vector<int64_t>& shape,
                                              const std::string& op_name) {
    auto fh = get_default_forge();

    if (op_name == "zeros") {
        return std::make_shared<ArrayHandle>(shape, fh->device_ptr(), /*zero=*/true);
    }

    auto out = std::make_shared<ArrayHandle>(shape, fh->device_ptr());
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)fh->queue_ptr();
    uint32_t seed = fh->get_seed();

    id<MTLComputePipelineState> pipeline = get_pipeline(op_name, METAL_SOURCE);
    id<MTLBuffer> bufOut = (__bridge id<MTLBuffer>)out->metal_buffer();

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd)
        throw std::runtime_error(
            "Metal Error: Failed to create command buffer. GPU might out of memory.");
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) throw std::runtime_error("Metal Error: Failed to create command encoder.");
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufOut offset:0 atIndex:0];
    [enc setBytes:&seed length:4 atIndex:1];

    size_t numel = numel_from_shape(shape);
    MTLSize grid = MTLSizeMake(numel, 1, 1);
    MTLSize threads = MTLSizeMake(256, 1, 1);
    if (threads.width > grid.width) threads.width = grid.width;
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
    [enc endEncoding];

    [cmd commit];
    fh->set_seed(seed + (uint32_t)numel);
    out->set_event((__bridge void*)cmd);
    return out;
}
