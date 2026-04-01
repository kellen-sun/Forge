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

#define UNARY_OP(NAME1, OP1) \
kernel void NAME1( \
    const device float* A       [[ buffer(0) ]], \
    device float* Out           [[ buffer(1) ]], \
    constant long* shape        [[ buffer(2) ]], \
    constant long* strides_A    [[ buffer(3) ]], \
    constant long& offset_A     [[ buffer(4) ]], \
    constant uint& ndim         [[ buffer(5) ]], \
    \
    uint gid                    [[ thread_position_in_grid ]]) \
{ \
    /* Calculate Read Locations */ \
    uint idx_a = get_strided_index(gid, shape, strides_A, offset_A, ndim); \
    \
    Out[gid] = OP1(A[idx_a]); \
}
UNARY_OP(exp, exp)
UNARY_OP(exp2, exp2)
UNARY_OP(exp10, exp10)
UNARY_OP(log, log)
UNARY_OP(log2, log2)
UNARY_OP(log10, log10)
UNARY_OP(sqrt, sqrt)
UNARY_OP(rsqrt, rsqrt)
UNARY_OP(abs, abs)
UNARY_OP(sign, sign)
UNARY_OP(ceil, ceil)
UNARY_OP(floor, floor)
UNARY_OP(round, round)
UNARY_OP(trunc, trunc)
UNARY_OP(fract, fract)
UNARY_OP(sin, sin)
UNARY_OP(cos, cos)
UNARY_OP(tan, tan)
UNARY_OP(asin, asin)
UNARY_OP(acos, acos)
UNARY_OP(atan, atan)
UNARY_OP(sinh, sinh)
UNARY_OP(cosh, cosh)
UNARY_OP(tanh, tanh)


#define BINARY_OP(NAME2, OP2) \
kernel void NAME2( \
    const device float* A       [[ buffer(0) ]], \
    const device float* B       [[ buffer(1) ]], \
    device float* Out           [[ buffer(2) ]], \
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
    Out[gid] = A[idx_a] OP2 B[idx_b]; \
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
