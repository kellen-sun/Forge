#include <metal_stdlib>
using namespace metal;

kernel void metal_sum_setup(
    device float* a [[ buffer(3) ]],
    device float* b [[ buffer(4) ]],
    device float* tempA [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]]
) {
    tempA[id] = a[id] * b[id];
}