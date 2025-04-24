#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device float* out [[ buffer(0) ]],
    device float* tempA [[ buffer(1) ]],
    device float* tempB [[ buffer(2) ]],
    device float* a [[ buffer(4) ]],
    device float* b [[ buffer(5) ]],
    uint id [[ thread_position_in_grid ]]
) {
    out[id] = out[id];
}