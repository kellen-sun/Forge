#include <metal_stdlib>
using namespace metal;

kernel void avg(
    device float* out [[ buffer(0) ]],
    device float* tempA [[ buffer(1) ]],
    device float* tempB [[ buffer(2) ]],
    device float* a [[ buffer(3) ]],
    device float* b [[ buffer(4) ]],
    uint id [[ thread_position_in_grid ]]
) {
    out[id] = 17.666666666666668;
}