#include <metal_stdlib>
using namespace metal;

kernel void matmult_kernel(
    constant uint3 &dims [[ buffer(3) ]], // dims.x = M, dims.y = N, dims.z = K
    device const float* A [[ buffer(1) ]],
    device const float* B [[ buffer(2) ]],
    device float* C [[ buffer(0) ]],
    uint2 gid [[ thread_position_in_grid ]]
) {
    uint M = dims.x;
    uint N = dims.y;
    uint K = dims.z;

    uint row = gid.y;
    uint col = gid.x;

    if (row >= M || col >= N) return;

    float acc = 0.0;
    for (uint k = 0; k < K; ++k) {
        float a = A[row * K + k];
        float b = B[k * N + col];
        acc += a * b;
    }

    C[row * N + col] = acc;
}
