#import <Metal/Metal.h>

#include "../include/array_elementwise.h"
#include "../include/metal_source.h"
#include "../include/metal_utils.h"

std::shared_ptr<ArrayHandle> array_unaryops(const std::shared_ptr<ArrayHandle>& A,
                                            const std::string& op_name) {
    return launch_elementwise(op_name, A->shape(), {A}, true);
}

std::shared_ptr<ArrayHandle> array_binops(const std::shared_ptr<ArrayHandle>& A,
                                          const std::shared_ptr<ArrayHandle>& B,
                                          const std::string& op_name) {
    const auto& shapeA = A->shape();
    const auto& shapeB = B->shape();
    std::vector<int64_t> out_shape = (shapeA == shapeB) ? shapeA : broadcast_shapes(shapeA, shapeB);

    return launch_elementwise(op_name, out_shape, {A, B}, true);
}

std::shared_ptr<ArrayHandle> array_inplaceops(const std::shared_ptr<ArrayHandle>& A,
                                              const std::shared_ptr<ArrayHandle>& B,
                                              const std::string& op_name) {
    const auto& shapeA = A->shape();
    const auto& shapeB = B->shape();
    std::vector<int64_t> out_shape = broadcast_shapes(shapeA, shapeB);
    if (out_shape != shapeA) throw std::runtime_error("array_inplaceops: broadcast failed");

    return launch_elementwise(op_name, shapeA, {A, B}, false);
}

// Nullary ops (rand/randn/zeros) don't share the same buffer layout
// so kept as a dedicated launcher
std::shared_ptr<ArrayHandle> array_nullaryops(const std::vector<int64_t>& shape,
                                              const std::string& op_name) {
    auto fh = get_default_forge();

    if (op_name == "zeros") {
        return std::make_shared<ArrayHandle>(shape, fh->device_ptr(), /*zero=*/true);
    }

    auto out = std::make_shared<ArrayHandle>(shape, fh->device_ptr());
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)fh->queue_ptr();
    uint32_t seed = fh->get_seed();

    id<MTLComputePipelineState> pipeline =
        (__bridge_transfer id<MTLComputePipelineState>)get_pipeline(op_name, METAL_SOURCE);
    id<MTLBuffer> bufOut = out->metal_buffer();

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
    out->set_event(cmd);
    return out;
}
