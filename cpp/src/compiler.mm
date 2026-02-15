#import <Metal/Metal.h>

#include "../include/compiler.h"
#include "../include/forge_handle.h"

void compile_metal(Graph& graph) {
    // auto defaultForgeHandle = get_default_forge();
    id<MTLDevice> device = (__bridge id<MTLDevice>)get_default_forge()->device_ptr();
    NSString* source = [NSString stringWithUTF8String:graph.shader_source.c_str()];
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.fastMathEnabled = YES;
    NSError* err = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&err];
    if (err) {
        NSLog(@"Library compilation failed: %@", [err localizedDescription]);
        if (!library) {
            throw std::runtime_error("Metal Error: Failed to compile Metal.");
        }
    }
    graph.pipeline = (__bridge_retained void*)library;
}

Graph::~Graph() {
    if (pipeline) {
        id old_pipeline = (__bridge_transfer id)pipeline;
        pipeline = nullptr;
    }
}
