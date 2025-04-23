import inspect
import ast
import Metal
import numpy as np
from transpiler import transpile, get_args, get_func_args


def metal(func):
    source = inspect.getsource(func)
    tree = ast.parse(source)
    
    metal_args = get_args(tree)
    func_args = get_func_args(tree)

    def wrapper(*args, **kwargs):
        device = Metal.MTLCreateSystemDefaultDevice()

        np_inputs = [np.zeros_like(args[0]),np.zeros_like(args[0]),np.zeros_like(args[0])]
        for i in range(len(func_args)):
            np_inputs.append(args[i])
        # For now we assume all inputs are numpy arrays of same type
        for i in range(len(metal_args)):
            if metal_args[i] not in func_args:
                np_inputs.append(np.zeros_like(args[0]))
        n = np_inputs[0].size

        # Allocate Metal buffers
        bufs = [device.newBufferWithLength_options_(
                    np_inputs[0].nbytes, 0
                ), 
                device.newBufferWithLength_options_(
                    np_inputs[1].nbytes, 0
                ),
                device.newBufferWithLength_options_(
                    np_inputs[2].nbytes, 0
                )]
        for i in range(len(func_args)):
            bufs.append(device.newBufferWithBytes_length_options_(
                np_inputs[i+3].tobytes(), np_inputs[i].nbytes, 0
            ))
        for i in range(len(metal_args)):
            if metal_args[i] not in func_args:
                bufs.append(device.newBufferWithLength_options_(
                    np_inputs[i+3].nbytes, 0
                ))

        source_str = transpile(tree, source, (device, bufs, n))
        options = None
        lib, error = device.newLibraryWithSource_options_error_(source_str, options, None)
        if lib is None:
            print("Failed to load Metal library:", error)
        fn = lib.newFunctionWithName_(func.__name__)

        # Create pipeline
        pipeline_state, _ = device.newComputePipelineStateWithFunction_error_(fn, None)

        # Create command queue + buffer
        cmd_queue = device.newCommandQueue()
        cmd_buf = cmd_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()

        # Set up encoder
        encoder.setComputePipelineState_(pipeline_state)
        for i, buf in enumerate(bufs):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)

        # Launch thread groups
        threads_per_group = Metal.MTLSizeMake(32, 1, 1)
        num_threadgroups = Metal.MTLSizeMake((n + 31) // 32, 1, 1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_threadgroups, threads_per_group)

        # Finalize and commit
        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        # Access the raw memory from the Metal buffer
        raw = bufs[0].contents()[:4*n]
        data_bytes = b''.join(raw)  # Flatten to a single bytes object
        result = np.frombuffer(data_bytes, dtype=np.float32)
        return result

    return wrapper

