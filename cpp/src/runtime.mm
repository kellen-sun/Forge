#import <Metal/Metal.h>
#include "../include/compiler.h"
#include "../include/forge_handle.h"
#include "../include/graph.h"
#include "../include/memory_arena.h"

std::shared_ptr<ArrayHandle> Graph::execute(std::vector<std::shared_ptr<ArrayHandle>> inputs) {
    // Combine graph with inputs to execute and produce the output
    // No need to copy inputs over, we just edit in place if needed, because its pass-by-ref
    auto defaultForgeHandle = get_default_forge();
    id<MTLDevice> device = (__bridge id<MTLDevice>)defaultForgeHandle->device_ptr();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)defaultForgeHandle->queue_ptr();
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    // a) Allocate the memory plan needed
    // i. allocate the arena
    uint64_t total_bytes = this->arena->get_total_bytes();
    id<MTLBuffer> arena_buffer = nil;
    if (total_bytes > 0) {
        arena_buffer = [device newBufferWithLength:total_bytes
                                           options:MTLResourceStorageModeShared];
    }
    // a) ii. Allocate the output ArrayHandle (not part of Arena to allow Arena to be freed)
    int output_root = this->arena->get_root(this->output_index);
    std::shared_ptr<ArrayHandle> root_handle;
    if (this->nodes[output_root].op == OpCode::INPUT) {
        // if the output is just a view of the input
        root_handle = inputs[output_root];
    } else {
        root_handle = std::make_shared<ArrayHandle>(this->nodes[output_root].shape);
    }
    std::shared_ptr<ArrayHandle> output_handle;
    if (this->output_index == output_root) {
        output_handle = root_handle;
    } else {
        output_handle = std::make_shared<ArrayHandle>(
            root_handle, this->nodes[this->output_index].shape,
            this->nodes[this->output_index].strides, this->nodes[this->output_index].offset);
    }

    // Helper to find which buffers a node's I/O refers to: (inputHandle, Arena or outputHandle)
    auto get_metal_buffer_for_node = [&](int node_idx) -> id<MTLBuffer> {
        int root = this->arena->get_root(node_idx);
        if (root == output_root) return output_handle->metal_buffer();
        if (this->nodes[root].op == OpCode::INPUT) {
            return inputs[root]->metal_buffer();
        }
        return arena_buffer;
    };

    for (size_t i = 0; i < this->nodes.size(); ++i) {
        const Node& node = this->nodes[i];
        const KernelConfig& config = this->configs[i];
        void* raw_ptr = this->pipelines[i];
        // config.name.empty() IFF pipeline == nullptr
        // TODO: add an assert (?) but there's other places where we should assert
        // like if .size() are equal needs to be asserted somewhere probably
        if (!raw_ptr) continue;

        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)raw_ptr;
        [computeEncoder setComputePipelineState:pso];
        // b) Set the correct buffer inputs
        id<MTLBuffer> dest_buf = get_metal_buffer_for_node(i);
        uint64_t dest_offset = this->arena->get_offset(i) + (node.offset * sizeof(float));
        [computeEncoder setBuffer:dest_buf offset:dest_offset atIndex:0];
        int bind_index = 1;
        for (int in_idx : node.inputs) {
            id<MTLBuffer> in_buf = get_metal_buffer_for_node(in_idx);
            uint64_t in_offset =
                this->arena->get_offset(in_idx) + (this->nodes[in_idx].offset * sizeof(float));

            [computeEncoder setBuffer:in_buf offset:in_offset atIndex:bind_index++];
        }

        // c) Launch the kernels one by one
        MTLSize grid = MTLSizeMake(config.grid[0], config.grid[1], config.grid[2]);
        MTLSize threadgroup = MTLSizeMake(config.group[0], config.group[1], config.group[2]);
        [computeEncoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
        // TODO: Noting when synchs are necessary; we could always synch between two kernels
        // But technically only need to synch if RAW and WAW cases
        // (like when they share the same mem in the arena)
        // -> Change to: id<MTLComputeCommandEncoder> computeEncoder =
        // [commandBuffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        // And add fences when needed: MTLFence
    }
    [computeEncoder endEncoding];
    [commandBuffer commit];
    output_handle->set_event(commandBuffer);
    return output_handle;
}
