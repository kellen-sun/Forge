#include "../include/metal_utils.h"
#include <iostream>
#include <map>
#include "../include/array_handle.h"
#include "../include/metal_source.h"

id<MTLComputePipelineState> get_pipeline(const std::string& op_name, const char* metal_c_string) {
    static std::map<std::string, id<MTLComputePipelineState>> cache;
    static id<MTLLibrary> library = nil;
    auto defaultForgeHandle = get_default_forge();
    id<MTLDevice> device = (__bridge id<MTLDevice>)defaultForgeHandle->device_ptr();

    if (!library) {
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
