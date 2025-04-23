#include <metal_stdlib>
using namespace metal;

kernel void metal_sum_setup(
    device float* b [[ buffer(4) ]],
    device float* tempA [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]]
) {
    tempA[id] = 14.0 * b[id];
}