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
}

// Batched matrix multiplication kernel
// Computes C[b, i, j] = sum_k A[b, i, k] * B[b, k, j]
// Uses tiled approach with threadgroup memory for better performance
constant uint TILE_SIZE = 16;

kernel void batched_matmul(
    const device float* A       [[ buffer(0) ]],
    const device float* B       [[ buffer(1) ]],
    device float* C             [[ buffer(2) ]],
    constant long& M            [[ buffer(3) ]],  // rows of A, rows of C
    constant long& K            [[ buffer(4) ]],  // cols of A, rows of B
    constant long& N            [[ buffer(5) ]],  // cols of B, cols of C
    constant long& batch_size   [[ buffer(6) ]],
    constant long& stride_a     [[ buffer(7) ]],  // batch stride for A (M*K)
    constant long& stride_b     [[ buffer(8) ]],  // batch stride for B (K*N)
    constant long& stride_c     [[ buffer(9) ]],  // batch stride for C (M*N)
    uint3 gid                   [[ thread_position_in_grid ]],
    uint3 tid                   [[ thread_position_in_threadgroup ]],
    uint3 tgid                  [[ threadgroup_position_in_grid ]])
{
    // gid.z = batch index, gid.y = row tile, gid.x = col
    uint batch = gid.z;
    uint row = gid.y;
    uint col = gid.x;

    if (batch >= (uint)batch_size || row >= (uint)M || col >= (uint)N) {
        return;
    }

    // Calculate base pointers for this batch
    const device float* A_batch = A + batch * stride_a;
    const device float* B_batch = B + batch * stride_b;
    device float* C_batch = C + batch * stride_c;

    // Compute dot product for C[row, col]
    float sum = 0.0f;
    for (uint k = 0; k < (uint)K; ++k) {
        sum += A_batch[row * K + k] * B_batch[k * N + col];
    }

    C_batch[row * N + col] = sum;
}

// Tiled batched matmul for better cache utilization
kernel void batched_matmul_tiled(
    const device float* A       [[ buffer(0) ]],
    const device float* B       [[ buffer(1) ]],
    device float* C             [[ buffer(2) ]],
    constant long& M            [[ buffer(3) ]],
    constant long& K            [[ buffer(4) ]],
    constant long& N            [[ buffer(5) ]],
    constant long& batch_size   [[ buffer(6) ]],
    constant long& stride_a     [[ buffer(7) ]],
    constant long& stride_b     [[ buffer(8) ]],
    constant long& stride_c     [[ buffer(9) ]],
    uint3 gid                   [[ thread_position_in_grid ]],
    uint3 tid                   [[ thread_position_in_threadgroup ]],
    uint3 tgid                  [[ threadgroup_position_in_grid ]])
{
    // Shared memory tiles
    threadgroup float A_tile[TILE_SIZE][TILE_SIZE];
    threadgroup float B_tile[TILE_SIZE][TILE_SIZE];

    uint batch = tgid.z;
    uint row = tgid.y * TILE_SIZE + tid.y;
    uint col = tgid.x * TILE_SIZE + tid.x;

    if (batch >= (uint)batch_size) return;

    const device float* A_batch = A + batch * stride_a;
    const device float* B_batch = B + batch * stride_b;
    device float* C_batch = C + batch * stride_c;

    float sum = 0.0f;

    // Loop over tiles
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (uint t = 0; t < num_tiles; ++t) {
        // Load A tile
        uint a_col = t * TILE_SIZE + tid.x;
        if (row < (uint)M && a_col < (uint)K) {
            A_tile[tid.y][tid.x] = A_batch[row * K + a_col];
        } else {
            A_tile[tid.y][tid.x] = 0.0f;
        }

        // Load B tile
        uint b_row = t * TILE_SIZE + tid.y;
        if (b_row < (uint)K && col < (uint)N) {
            B_tile[tid.y][tid.x] = B_batch[b_row * N + col];
        } else {
            B_tile[tid.y][tid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute partial dot product
        for (uint k = 0; k < TILE_SIZE; ++k) {
            sum += A_tile[tid.y][k] * B_tile[k][tid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (row < (uint)M && col < (uint)N) {
        C_batch[row * N + col] = sum;
    }
})";
