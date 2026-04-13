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

inline uint hash(uint seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

inline float rand_uniform(uint gid, uint base_seed) {
    uint random_int = hash(base_seed + gid);

    return float(random_int) / 4294967295.0;
}

inline float rand_normal(uint gid, uint base_seed) {
    float u1 = max(rand_uniform(gid * 2, base_seed), 1e-7f);
    float u2 = rand_uniform(gid * 2 + 1, base_seed);

    float r = sqrt(-2.0 * log(u1));
    float theta = 2.0 * 3.14159265359 * u2;

    return r * cos(theta);
}

#define NULLARY_OP(NAME0, OP0) \
kernel void NAME0( \
    device float* Out           [[ buffer(0) ]], \
    constant uint& seed         [[ buffer(1) ]], \
    \
    uint gid                    [[ thread_position_in_grid ]]) \
{ \
    Out[gid] = OP0(gid, seed); \
}
NULLARY_OP(rand, rand_uniform)
NULLARY_OP(randn, rand_normal)

#define INPLACE_OP(NAME10, OP10) \
kernel void NAME10( \
    device float* A       [[ buffer(0) ]], \
    const device float* B       [[ buffer(1) ]], \
    constant long* shape        [[ buffer(2) ]], \
    constant long* strides_A    [[ buffer(3) ]], \
    constant long& offset_A     [[ buffer(4) ]], \
    constant long* strides_B    [[ buffer(5) ]], \
    constant long& offset_B     [[ buffer(6) ]], \
    constant uint& ndim         [[ buffer(7) ]], \
    \
    uint gid                    [[ thread_position_in_grid ]]) \
{ \
    /* Calculate Read Locations */ \
    uint idx_a = get_strided_index(gid, shape, strides_A, offset_A, ndim); \
    uint idx_b = get_strided_index(gid, shape, strides_B, offset_B, ndim); \
    \
    A[idx_a] = A[idx_a] OP10 B[idx_b]; \
}
INPLACE_OP(iadd, +)
INPLACE_OP(isub, -)
INPLACE_OP(imul, *)
INPLACE_OP(idiv, /)

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
