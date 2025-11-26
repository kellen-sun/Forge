#import <Metal/Metal.h>

#include "../include/array_add.h"

enum class ArrayOperationType : int {
    ADD = 0,
    SUB = 1,
    MULT = 2,
    DIV = 3
}

std::shared_ptr<ArrayHandle> operations_arrays_cpp(
    const std::shared_ptr<ArrayHandle>& A, 
    const std::shared_ptr<ArrayHandle>& B,
    ArrayOperationType op_type) //parameter for operation type
{
    const auto& shapeA = A->shape();
    const auto& shapeB = B->shape();

    if (shapeA != shapeB) {
        throw std::runtime_error("operations_arrays_cpp: shape mismatch");
    }

    auto defaultForgeHandle = get_default_forge();
    id<MTLDevice> device =  (__bridge id<MTLDevice>) defaultForgeHandle->device_ptr();
    id<MTLCommandQueue> queue =  (__bridge id<MTLCommandQueue>) defaultForgeHandle->queue_ptr();

    // compile pipeline on first call
    static id<MTLComputePipelineState> pipeline = nil;
    if (!pipeline) {
        NSString* source = loadMetalSource("elementwise.metal");
        if (!source ){
            throw std::runtime_error("Failed to load Metal source file.")
        }
        id<MTLLibrary> lib = [device newLibraryWithSource:source options:nil error:nil];
        id<MTLFunction> fn = [lib newFunctionWithName:@"operations_arrays"];
        pipeline = [device newComputePipelineStateWithFunction:fn error:nil];
    }

    // allocate output ArrayHandle
    auto out = std::make_shared<ArrayHandle>(
        A->shape(),
        defaultForgeHandle->device_ptr()
    );
    
    id<MTLBuffer> bufA = (__bridge id<MTLBuffer>) A->metal_buffer();
    id<MTLBuffer> bufB = (__bridge id<MTLBuffer>) B->metal_buffer();
    id<MTLBuffer> bufOut = (__bridge id<MTLBuffer>) out->metal_buffer();

    const int op_code = static_cast<int>(op_type);

    id<MTLBuffer> op_type_buffer = [device newBufferWithBytes:&op_code
                                        length: sizeof(int) 
                                        options:MTLResourceStorageModeManaged];

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd)
        throw std::runtime_error("Metal Error: Failed to create command buffer. GPU might out of memory.");
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc)
        throw std::runtime_error("Metal Error: Failed to create command encoder.");
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufOut offset:0 atIndex:2];
    [enc setBuffer:op_type_buffer offset:0 atIndex:3];

    MTLSize grid = MTLSizeMake(A->data().size(), 1, 1);
    MTLSize threads = MTLSizeMake(256, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
    [enc endEncoding];

    [cmd commit];
    out->set_event((__bridge void*)cmd);

    return out;
}
