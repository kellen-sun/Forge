#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(
    device float* a [[ buffer(0) ]],
    device float* b [[ buffer(1) ]],
    device float* x [[ buffer(2) ]],
    device float* y [[ buffer(3) ]],
    device float* out [[ buffer(4) ]],
    uint id [[ thread_position_in_grid ]]
) {
    x[id] = a[id] + b[id];
    y[id] = a[id] * 2;
    out[id] = x[id] * y[id];
}