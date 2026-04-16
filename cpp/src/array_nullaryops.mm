#import <Metal/Metal.h>

#include "../include/array_nullaryops.h"
#include "../include/metal_source.h"
#include "../include/metal_utils.h"

std::shared_ptr<ArrayHandle> array_nullaryops(const std::vector<int64_t>& shape,
                                              const std::string& op_name) {
    auto defaultForgeHandle = get_default_forge();

    // allocate output ArrayHandle
    if (op_name == "zeros") {
        return std::make_shared<ArrayHandle>(shape, defaultForgeHandle->device_ptr(), true);
    }
    auto out = std::make_shared<ArrayHandle>(shape, defaultForgeHandle->device_ptr());

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)defaultForgeHandle->queue_ptr();
    uint32_t seed = defaultForgeHandle->get_seed();

    // compile pipeline on first call
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

    MTLSize grid = MTLSizeMake(numel_from_shape(shape), 1, 1);
    MTLSize threads = MTLSizeMake(256, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
    [enc endEncoding];

    [cmd commit];
    defaultForgeHandle->set_seed(seed + (uint32_t)numel_from_shape(shape));
    out->set_event((__bridge void*)cmd);

    return out;
}
