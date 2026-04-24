#pragma once
#include <cstdint>
#include <initializer_list>
#include <string>

#include "array_handle.h"

void* get_pipeline(const std::string& op_name, const char* metal_c_string);

// Launches a kernel that follows the shared elementwise buffer layout:
//   inputs[0..N-1]              at slots 0..N-1
//   out_buf (if non-nil)        at slot N
//   shape                       next
//   (strides_i, offset_i) pairs for each input, in order
//   ndim                        last
// Each input's strides are broadcast to out_shape via get_bcast_strides.
// Commits the command buffer and returns it so the caller can wire
// set_event on the result handle.
std::shared_ptr<ArrayHandle> launch_elementwise(
    const std::string& op_name, const std::vector<int64_t>& out_shape,
    std::initializer_list<const std::shared_ptr<ArrayHandle>> inputs, bool dedicated_out);
