#import <Metal/Metal.h>

#include "../include/array_add.h"

std::shared_ptr<ArrayHandle> add_arrays_cpp(const std::shared_ptr<ArrayHandle>& A, 
    const std::shared_ptr<ArrayHandle>& B) 
{
    const auto& shapeA = A->shape();
    const auto& shapeB = B->shape();

    if (shapeA != shapeB) {
        throw std::runtime_error("add_arrays_cpp: shape mismatch");
    }

    auto defaultForgeHandle = get_default_forge();
    id<MTLDevice> device =  (__bridge id<MTLDevice>) defaultForgeHandle->device_ptr();
    id<MTLCommandQueue> queue =  (__bridge id<MTLCommandQueue>) defaultForgeHandle->queue_ptr();

    // compile pipeline on first call
    static id<MTLComputePipelineState> pipeline = nil;
    if (!pipeline) {
        NSString* source = @R"(#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(
    const device float* A       [[ buffer(0) ]],
    const device float* B       [[ buffer(1) ]],
    device float* Out           [[ buffer(2) ]],
    uint gid                    [[ thread_position_in_grid ]]
)
{
    Out[gid] = A[gid] + B[gid];
}
)";
        id<MTLLibrary> lib = [device newLibraryWithSource:source options:nil error:nil];
        id<MTLFunction> fn = [lib newFunctionWithName:@"add_arrays"];
        pipeline = [device newComputePipelineStateWithFunction:fn error:nil];
    }

    // allocate output ArrayHandle
    auto out = std::make_shared<ArrayHandle>(
        std::vector<float>(A->data().size(), 0.0f),
        A->shape(),
        get_default_forge()->device_ptr()
    );
    
    id<MTLBuffer> bufA = (__bridge id<MTLBuffer>) A->metal_buffer();
    id<MTLBuffer> bufB = (__bridge id<MTLBuffer>) B->metal_buffer();
    id<MTLBuffer> bufOut = (__bridge id<MTLBuffer>) out->metal_buffer();

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufOut offset:0 atIndex:2];

    MTLSize grid = MTLSizeMake(A->data().size(), 1, 1);
    MTLSize threads = MTLSizeMake(256, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
    [enc endEncoding];

    [cmd commit];
    [cmd waitUntilCompleted];

    return out;
}