#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(device float *a, device float *b, device float *out, uint id [[thread_position_in_grid]]) {
    out[id] = a[id] + b[id];
}
