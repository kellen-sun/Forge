#pragma once
#import <Metal/Metal.h>

#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>

id<MTLComputePipelineState> get_pipeline(const std::string& op_name, const char* metal_c_string);

// One strided input to an elementwise kernel. Its shape/strides are broadcast
// up to the caller-provided out_shape inside launch_elementwise.
struct StridedInput {
    id<MTLBuffer> buf;
    std::span<const int64_t> shape;
    std::span<const int64_t> strides;
    size_t offset;
};

// Launches a kernel that follows the shared elementwise buffer layout:
//   inputs[0..N-1]              at slots 0..N-1
//   out_buf (if non-nil)        at slot N
//   shape                       next
//   (strides_i, offset_i) pairs for each input, in order
//   ndim                        last
// Each input's strides are broadcast to out_shape via get_bcast_strides.
// Commits the command buffer and returns it so the caller can wire
// set_event on the result handle.
id<MTLCommandBuffer> launch_elementwise(const std::string& op_name,
                                        std::span<const int64_t> out_shape,
                                        std::initializer_list<StridedInput> inputs,
                                        id<MTLBuffer> out_buf);
