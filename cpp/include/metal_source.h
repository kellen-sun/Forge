#pragma once

#include <string>

const char* const ELEMENTWISE_METAL_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

uint get_strided_index(uint gid, 
                       constant long* shape, 
                       constant long* strides, 
                       constant long& offset, 
                       uint dims) 
{
    uint physical_idx = offset;
    uint remaining = gid;

    // Unravel 'gid' into N-D coordinates
    // We iterate backwards (from last dim to first)
    for (int i = dims - 1; i >= 0; --i) {
        uint coordinate = remaining % shape[i]; 
        physical_idx += coordinate * strides[i]; 
        remaining /= shape[i]; 
    }
    return physical_idx;
}

// The Strided Macro
#define BINARY_OP(NAME, OP) \
kernel void NAME( \
    const device float* A       [[ buffer(0) ]], \
    const device float* B       [[ buffer(1) ]], \
    device float* Out           [[ buffer(2) ]], \
    \
    constant long* shape        [[ buffer(3) ]], /* Iteration Space */ \
    constant long* strides_A    [[ buffer(4) ]], \
    constant long& offset_A     [[ buffer(5) ]], \
    constant long* strides_B    [[ buffer(6) ]], \
    constant long& offset_B     [[ buffer(7) ]], \
    constant uint& ndim         [[ buffer(8) ]], \
    \
    uint gid                    [[ thread_position_in_grid ]]) \
{ \
    /* Calculate Read Locations */ \
    uint idx_a = get_strided_index(gid, shape, strides_A, offset_A, ndim); \
    uint idx_b = get_strided_index(gid, shape, strides_B, offset_B, ndim); \
    \
    /* Math */ \
    Out[gid] = A[idx_a] OP B[idx_b]; \
}
BINARY_OP(add, +)
BINARY_OP(sub, -)
BINARY_OP(mul, *)
BINARY_OP(div, /)
)";