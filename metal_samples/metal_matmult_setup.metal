#include <metal_stdlib>
using namespace metal;

kernel void metal_matmult_setup(
    device float* b [[ buffer(5) ]],
    device float* out [[ buffer(0) ]],
    uint id [[ thread_position_in_grid ]]
) {
    out[id] = b[id];
}