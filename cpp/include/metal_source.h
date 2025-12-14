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

    for (int i = dims - 1; i >= 0; --i) {
        uint coordinate = remaining % shape[i]; 
        physical_idx += coordinate * strides[i]; 
        remaining /= shape[i]; 
    }
    return physical_idx;
}

#define BINARY_OP(NAME, OP) \
kernel void NAME( \
    const device float* A       [[ buffer(0) ]], \
    const device float* B       [[ buffer(1) ]], \
    device float* Out           [[ buffer(2) ]], \
    \
    constant long* shape        [[ buffer(3) ]], \
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
    Out[gid] = A[idx_a] OP B[idx_b]; \
}
BINARY_OP(add, +)
BINARY_OP(sub, -)
BINARY_OP(mul, *)
BINARY_OP(div, /)

kernel void copy_view(
    device float* Dest          [[ buffer(0) ]],
    const device float* Src     [[ buffer(1) ]],
    constant long* shape        [[ buffer(2) ]],
    constant long* strides_dst  [[ buffer(3) ]],
    constant long& offset_dst   [[ buffer(4) ]],
    constant long* strides_src  [[ buffer(5) ]],
    constant long& offset_src   [[ buffer(6) ]],
    constant uint& ndim         [[ buffer(7) ]],
    uint gid                    [[ thread_position_in_grid ]])
{
    uint idx_dst = get_strided_index(gid, shape, strides_dst, offset_dst, ndim);

    uint idx_src = get_strided_index(gid, shape, strides_src, offset_src, ndim);

    Dest[idx_dst] = Src[idx_src];
})";