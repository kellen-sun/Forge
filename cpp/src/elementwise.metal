#include <metal_stdlib>
using namespace metal;

kernel void add_arrays(
    const device float* A       [[ buffer(0) ]],
    const device float* B       [[ buffer(1) ]],
    device float* Out           [[ buffer(2) ]],
    constant int& op_type       [[ buffer(3) ]],
    uint gid                    [[ thread_position_in_grid ]]
)
{
    float result;

    switch (op_type) { 
        case 0:
            result = A[gid] + B[gid];
            break;
        case 1:
            result = A[gid] - B[gid];
            break;
        case 2:
            result = A[gid] * B[gid];
            break;
        case 3:
            result = A[gid] / B[gid];
            break;
        default:
            result = A[gid] + B[gid];
            break;
    }

    Out[gid] = result;
    
}
