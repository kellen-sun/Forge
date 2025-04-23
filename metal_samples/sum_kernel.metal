#include <metal_stdlib>
using namespace metal;

kernel void sum_kernel(
    device float* a [[ buffer(1) ]],
    device float* partials [[ buffer(0) ]],
    uint id [[ thread_position_in_grid ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint tg [[ threadgroup_position_in_grid ]]
) {
    threadgroup float shared[32];
    float val = a[id];
    shared[tid] = val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Simple reduction for 32 threads
    for (uint stride = 16; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        partials[tg] = shared[0];
    }
}
