#include "../include/metal_utils.h"

#include <iostream>
#include <map>
#include <vector>

#include "../include/array_handle.h"
#include "../include/broadcast.h"
#include "../include/metal_source.h"

id<MTLComputePipelineState> get_pipeline(const std::string& op_name, const char* metal_c_string) {
    static std::map<std::string, id<MTLComputePipelineState>> cache;
    static id<MTLLibrary> library = nil;
    auto defaultForgeHandle = get_default_forge();
    id<MTLDevice> device = (__bridge id<MTLDevice>)defaultForgeHandle->device_ptr();

    if (!library) {
        NSString* source = [NSString stringWithUTF8String:metal_c_string];
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.mathMode = MTLMathModeFast;
        NSError* err = nil;
        library = [device newLibraryWithSource:source options:options error:&err];
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
        throw std::runtime_error("get_pipeline: Failed to find function '" + op_name +
                                 "' in library");
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn
                                                                                 error:nil];

    if (!pipeline) {
        throw std::runtime_error("Metal Error: Failed to create pipeline state for " + op_name);
    }
    cache[op_name] = pipeline;
    return pipeline;
}

id<MTLCommandBuffer> launch_elementwise(const std::string& op_name,
                                        std::span<const int64_t> out_shape,
                                        std::initializer_list<StridedInput> inputs,
                                        id<MTLBuffer> out_buf) {
    auto fh = get_default_forge();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)fh->queue_ptr();
    id<MTLComputePipelineState> pipeline = get_pipeline(op_name, METAL_SOURCE);

    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    if (!cmd)
        throw std::runtime_error(
            "Metal Error: Failed to create command buffer. GPU might out of memory.");
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (!enc) throw std::runtime_error("Metal Error: Failed to create command encoder.");
    [enc setComputePipelineState:pipeline];

    NSUInteger slot = 0;
    for (const auto& in : inputs) {
        [enc setBuffer:in.buf offset:0 atIndex:slot++];
    }
    if (out_buf) {
        [enc setBuffer:out_buf offset:0 atIndex:slot++];
    }

    // 0-d scalars are dispatched as a single-element 1-d kernel: shape=[1], stride=[0], ndim=1.
    uint ndim_raw = (uint)out_shape.size();
    uint ndim_safe = ndim_raw == 0 ? 1 : ndim_raw;
    uint64_t scalar_shape = 1;
    uint64_t scalar_stride = 0;

    if (ndim_raw == 0) {
        [enc setBytes:&scalar_shape length:8 atIndex:slot++];
    } else {
        [enc setBytes:out_shape.data() length:ndim_raw * 8 atIndex:slot++];
    }

    std::vector<std::vector<int64_t>> stride_store;
    stride_store.reserve(inputs.size());
    for (const auto& in : inputs) {
        stride_store.push_back(get_bcast_strides(in.shape, in.strides, out_shape));
        if (ndim_raw == 0) {
            [enc setBytes:&scalar_stride length:8 atIndex:slot++];
        } else {
            [enc setBytes:stride_store.back().data() length:ndim_raw * 8 atIndex:slot++];
        }
        size_t off = in.offset;
        [enc setBytes:&off length:sizeof(size_t) atIndex:slot++];
    }

    [enc setBytes:&ndim_safe length:4 atIndex:slot++];

    size_t numel = 1;
    for (int64_t d : out_shape) numel *= (size_t)d;
    MTLSize grid = MTLSizeMake(numel, 1, 1);
    MTLSize threads = MTLSizeMake(256, 1, 1);
    if (threads.width > grid.width) threads.width = grid.width;
    [enc dispatchThreads:grid threadsPerThreadgroup:threads];
    [enc endEncoding];

    [cmd commit];
    return cmd;
}
