#import <Metal/Metal.h>

#include "../include/array_binops.h"
#include "../include/metal_source.h"

static id<MTLComputePipelineState> get_pipeline(const std::string& op_name) {
    static std::map<std::string, id<MTLComputePipelineState>> cache;
    static id<MTLLibrary> library = nil;
    auto defaultForgeHandle = get_default_forge();
    id<MTLDevice> device =  (__bridge id<MTLDevice>) defaultForgeHandle->device_ptr();

    if (!library) {
        const char* metal_c_string = ELEMENTWISE_METAL_SOURCE;
        NSString* source = [NSString stringWithUTF8String:metal_c_string];
        NSError* err = nil;
        library = [device newLibraryWithSource:source options:nil error:&err];
        if (!library) {
            NSLog(@"Library compilation failed: %@", [err localizedDescription]);
            throw std::runtime_error("Metal Error: Failed to compile Metal.");
        }
    }

    if (cache.find(op_name) != cache.end()) {
        return cache[op_name];
    }
    NSString* nameNS = [NSString stringWithUTF8String:op_name.c_str()];
    id<MTLFunction> fn = [library newFunctionWithName:nameNS];
    if (!fn) {
        throw std::runtime_error("get_pipeline: Failed to find function '" + op_name + "' in library");
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:nil];
    
    if (!pipeline) {
        throw std::runtime_error("Metal Error: Failed to create pipeline state for " + op_name);
    }
    cache[op_name] = pipeline;
    return pipeline;
}

std::shared_ptr<ArrayHandle> binops_arrays_cpp(
    const std::shared_ptr<ArrayHandle>& A, 
    const std::shared_ptr<ArrayHandle>& B,
    const std::string& op_name)
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
    id<MTLComputePipelineState> pipeline = get_pipeline(op_name);

    // allocate output ArrayHandle
    auto out = std::make_shared<ArrayHandle>(
        A->shape(),
        defaultForgeHandle->device_ptr()
    );
    
    id<MTLBuffer> bufA = (__bridge id<MTLBuffer>) A->metal_buffer();
    id<MTLBuffer> bufB = (__bridge id<MTLBuffer>) B->metal_buffer();
    id<MTLBuffer> bufOut = (__bridge id<MTLBuffer>) out->metal_buffer();

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

    uint ndim = (uint)out->shape().size();

    [enc setBytes:out->shape().data() length:ndim*8             atIndex:3];
    size_t current_offsetA = A->offset();
    [enc setBytes:A->strides().data() length:ndim*8             atIndex:4];
    [enc setBytes:&current_offsetA    length:sizeof(size_t)     atIndex:5];
    size_t current_offsetB = B->offset();
    [enc setBytes:B->strides().data() length:ndim*8             atIndex:6];
    [enc setBytes:&current_offsetB    length:sizeof(size_t)     atIndex:7];

    [enc setBytes:&ndim               length:4                  atIndex:8];

    MTLSize grid = MTLSizeMake(A->data().size(), 1, 1);
    MTLSize threads = MTLSizeMake(256, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
    [enc endEncoding];

    [cmd commit];
    out->set_event((__bridge void*)cmd);

    return out;
}
