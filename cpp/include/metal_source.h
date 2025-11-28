#pragma once

#include <string>

const char* const ELEMENTWISE_METAL_SOURCE = R"(
#include <metal_stdlib>
using namespace metal;

kernel void operations_arrays(
    const device float* A       [[ buffer(0) ]],
    const device float* B       [[ buffer(1) ]],
    device float* Out           [[ buffer(2) ]],
    constant int& op_type       [[ buffer(3) ]], // 0: ADD, 1: SUB, 2: MULT, 3: DIV
    uint gid                    [[ thread_position_in_grid ]]
)
{
    float result;

    switch (op_type) { 
        case 0: // ADD
            result = A[gid] + B[gid];
            break;
        case 1: // SUBTRACTION
            result = A[gid] - B[gid];
            break;
        case 2: // MULTIPLICATION
            result = A[gid] * B[gid];
            break;
        case 3: // DIVISION
            result = A[gid] / B[gid];
            break;
        default:
            result = A[gid] + B[gid];
            break;
    }

    Out[gid] = result;
}
)";