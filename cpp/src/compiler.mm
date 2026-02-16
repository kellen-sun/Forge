#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../include/compiler.h"
#include "../include/forge_handle.h"

void compile_metal(Graph& graph) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)get_default_forge()->device_ptr();
    NSString* source = [NSString stringWithUTF8String:graph.shader_source.c_str()];
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.mathMode = MTLMathModeFast;
    NSError* err = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&err];
    if (err) {
        NSLog(@"Library compilation failed: %@", [err localizedDescription]);
        if (!library) {
            throw std::runtime_error("Metal Error: Failed to compile Metal.");
        }
    }
    for (auto& config : graph.configs) {
        if (config.name.empty()) {
            graph.pipelines.push_back(nullptr);
            continue;
        }
        NSString* fnName = [NSString stringWithUTF8String:config.name.c_str()];
        id<MTLFunction> fn = [library newFunctionWithName:fnName];
        if (!fn) {
            throw std::runtime_error("get_pipeline: Failed to find function '" + config.name +
                                     "' in library");
        }
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn
                                                                                     error:nil];

        if (!pipeline) {
            throw std::runtime_error("Metal Error: Failed to create pipeline state for " +
                                     config.name);
        }
        graph.pipelines.push_back((__bridge_retained void*)pipeline);
    }
}

Graph::~Graph() {
    for (auto& pipeline : pipelines) {
        if (pipeline) {
            id old_pipeline = (__bridge_transfer id)pipeline;
            pipeline = nullptr;
        }
    }
}
