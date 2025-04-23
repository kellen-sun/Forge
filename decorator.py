import inspect
import ast
import Metal
import numpy as np
from Foundation import NSURL
from transpiler import transpile



def metal(func):
    source = inspect.getsource(func)
    tree = ast.parse(source)
    # print(ast.dump(tree, indent=4))
    
    # Extract the operation in `return a + b`
    if True:
        return_stmt = next((n for n in ast.walk(tree) if isinstance(n, ast.Return)), None)
        func_name = func.__name__
        
        if isinstance(return_stmt.value, ast.BinOp) and isinstance(return_stmt.value.op, ast.Add):
            metal_op = "+"
        else:
            raise NotImplementedError("Only supports a + b for now")

        metal_code = f"""
#include <metal_stdlib>
using namespace metal;

kernel void {func_name}(
    device const float* a [[ buffer(0) ]],
    device const float* b [[ buffer(1) ]],
    device float* out [[ buffer(2) ]],
    uint id [[ thread_position_in_grid ]])
{{
    out[id] = a[id] {metal_op} b[id];
}}
""".strip()
    else:
        metal_code = transpile(tree, source)

    print("Generated Metal code:\n", metal_code)

    def wrapper(*args, **kwargs):
        device = Metal.MTLCreateSystemDefaultDevice()
        source_str = metal_code
        options = None
        lib, error = device.newLibraryWithSource_options_error_(source_str, options, None)
        if lib is None:
            print("Failed to load Metal library:", error)
        fn = lib.newFunctionWithName_(func_name)

        # Create pipeline
        pipeline_state, _ = device.newComputePipelineStateWithFunction_error_(fn, None)

        a_np = args[0]
        b_np = args[1]
        n = a_np.size
        out_np = np.zeros_like(a_np)

        # Allocate Metal buffers
        buf_a = device.newBufferWithBytes_length_options_(a_np.tobytes(), a_np.nbytes, 0)
        buf_b = device.newBufferWithBytes_length_options_(b_np.tobytes(), b_np.nbytes, 0)
        buf_out = device.newBufferWithLength_options_(out_np.nbytes, 0)

        # Create command queue + buffer
        cmd_queue = device.newCommandQueue()
        cmd_buf = cmd_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()

        # Set up encoder
        encoder.setComputePipelineState_(pipeline_state)
        encoder.setBuffer_offset_atIndex_(buf_a, 0, 0)
        encoder.setBuffer_offset_atIndex_(buf_b, 0, 1)
        encoder.setBuffer_offset_atIndex_(buf_out, 0, 2)

        # Launch thread groups
        threads_per_group = Metal.MTLSizeMake(32, 1, 1)
        num_threadgroups = Metal.MTLSizeMake((n + 31) // 32, 1, 1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_threadgroups, threads_per_group)

        # Finalize and commit
        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        # Access the raw memory from the Metal buffer
        raw = buf_out.contents()[:4*n]  # objc.varlist of 1-byte bytes objects
        data_bytes = b''.join(raw)  # Flatten to a single bytes object
        result = np.frombuffer(data_bytes, dtype=np.float32)
        return result

    return wrapper

