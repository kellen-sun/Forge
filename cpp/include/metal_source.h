#pragma once

#include <string>

const char* const ELEMENTWISE_METAL_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

#define BINARY_OP(NAME, OP) \
kernel void NAME( \
    const device float* A [[ buffer(0) ]], \
    const device float* B [[ buffer(1) ]], \
    device float* Out     [[ buffer(2) ]], \
    uint gid              [[ thread_position_in_grid ]]) \
{ \
    Out[gid] = A[gid] OP B[gid]; \
}
BINARY_OP(add, +)
BINARY_OP(sub, -)
BINARY_OP(mul, *)
BINARY_OP(div, /)
)";