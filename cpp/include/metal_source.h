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
    float a = A[gid];
    float b = B[gid];
    float result = 0.0f; // Initialize result to zero

    // --- Branchless Operation Selection using Masking ---
    
    // Create a mask (1.0f or 0.0f) for each operation type
    // The select function chooses between the second argument (1.0f) and the third argument (0.0f)
    // based on the boolean condition (op_type == N).

    // Mask for ADD (0)
    float mask_add = select(0.0f, 1.0f, op_type == 0); 
    result += mask_add * (a + b);

    // Mask for SUB (1)
    float mask_sub = select(0.0f, 1.0f, op_type == 1);
    result += mask_sub * (a - b);

    // Mask for MULT (2)
    float mask_mult = select(0.0f, 1.0f, op_type == 2);
    result += mask_mult * (a * b);

    // Mask for DIV (3)
    float mask_div = select(0.0f, 1.0f, op_type == 3);
    result += mask_div * (a / b);
    
    // --- Default/Error Handling ---
    // Since the C++ side ensures op_type is 0-3, the 'default' case is less critical.
    // If op_type is invalid (e.g., 99), all masks will be 0.0f, and result remains 0.0f.
    // If you strictly need to handle default, you would add an explicit mask for 
    // when NO other mask is 1.0f, but for performance, we rely on the host being correct.
    
    // --- Output ---
    Out[gid] = result;
}
)";