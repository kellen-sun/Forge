import transpiler
import ast
import Metal
from MetalPerformanceShaders import *
import numpy as np
from Foundation import NSURL

def overwrite_buffer(device_buffer, np_array):
    # Get raw pointer to buffer contents
    mv = device_buffer.contents().as_buffer(np_array.nbytes)
    # Copy the numpy array's raw bytes directly
    mv[:] = np_array.view(np.uint8)

class ReplaceSums(ast.NodeTransformer):
    def __init__(self, metal_sum_fn, buffer_lookup, runtime_items):
        self.metal_sum = metal_sum_fn
        self.buffer_lookup = buffer_lookup
        self.runtime_items = runtime_items

    def visit_Call(self, node):
        self.generic_visit(node)  # Recursively transform arguments
        if isinstance(node.func, ast.Name) and node.func.id == "sum":
            result = self.metal_sum(node, self.buffer_lookup, self.runtime_items)
            return ast.Constant(value=result)
        return node

class ReplaceMatMults(ast.NodeTransformer):
    def __init__(self, metal_matmul_fn, buffer_lookup, runtime_items):
        self.metal_matmul = metal_matmul_fn
        self.buffer_lookup = buffer_lookup
        self.runtime_items = runtime_items

    def visit_BinOp(self, node):
        self.generic_visit(node)  # Recursively transform arguments
        if isinstance(node.op, ast.MatMult):
            result = self.metal_matmul(node, self.buffer_lookup, self.runtime_items)
            return ast.Name(id=result)
        return node

# can do buffer 0 as output, 1 and 2 as temps
def metal_sum(node, buffer_lookup, runtime_items):
    device, bufs, n, cmd_queue = runtime_items
    func_name = "metal_sum_setup"
    decls = []
    transformer = ReplaceSums(metal_sum, buffer_lookup, runtime_items)
    new_node = transformer.visit(node.args[0])
    for N in ast.walk(new_node):
        if isinstance(N, ast.Name):
            if N.id not in decls:
                decls.append(N.id)
    buf_decls = "\n    ".join([
        f"device float* {arg} [[ buffer({buffer_lookup[arg]}) ]]," for arg in decls
    ])
    body = f"tempA[id] = {transpiler.get_expr(new_node)};"
    metal_sum_setup = f"""
#include <metal_stdlib>
using namespace metal;

kernel void {func_name}(
    {buf_decls}
    device float* tempA [[ buffer(1) ]],
    uint id [[ thread_position_in_grid ]]
) {{
    {body}
}}
""".strip()
    with open("./metal_samples/metal_sum_setup.metal", "w") as f:
        f.write(metal_sum_setup)
    
    # Compile AFTER buffers are set up
    lib, error = device.newLibraryWithSource_options_error_(metal_sum_setup, None, None)
    fn = lib.newFunctionWithName_(func_name)
    pipeline_state, _ = device.newComputePipelineStateWithFunction_error_(fn, None)

    # Setup and dispatch
    cmd_buf = cmd_queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()

    encoder.setComputePipelineState_(pipeline_state)
    for i, buf in enumerate(bufs):
        encoder.setBuffer_offset_atIndex_(buf, 0, i)

    # Run helper
    threads_per_group = Metal.MTLSizeMake(256, 1, 1)
    num_threadgroups = Metal.MTLSizeMake((n + 255) // 256, 1, 1)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_threadgroups, threads_per_group)

    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    # Load the metallib
    url = NSURL.fileURLWithPath_("./metal_samples/sum_kernel.metallib")
    lib, error = device.newLibraryWithURL_error_(url, None)
    if lib is None:
        print("Failed to load metallib:", error)
        return

    # Get the function + pipeline
    fn = lib.newFunctionWithName_("sum_kernel")
    pipeline_state, _ = device.newComputePipelineStateWithFunction_error_(fn, None)

    # Create new command buffer + encoder
    cmd_buf = cmd_queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()

    encoder.setComputePipelineState_(pipeline_state)

    encoder.setBuffer_offset_atIndex_(bufs[1], 0, 1)  # input
    encoder.setBuffer_offset_atIndex_(bufs[0], 0, 0)  # output

    # Launch with one threadgroup (for simplicity)
    threads_per_group = Metal.MTLSizeMake(256, 1, 1)
    num_threadgroups = Metal.MTLSizeMake(1, 1, 1)  # only 1 result

    encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_threadgroups, threads_per_group)
    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()

    count = num_threadgroups.width * num_threadgroups.height * num_threadgroups.depth
    ptr = bufs[0].contents().as_buffer(4 * count)
    partials = np.frombuffer(ptr, dtype=np.float32)

    sum_val = float(np.sum(partials))

    return sum_val

def metal_matmult(node, buffer_lookup, runtime_items):
    # return a string, which is the buffer name of output
    device, bufs, n, cmd_queue = runtime_items
    func_name = "metal_matmult_setup"
    transformer = ReplaceMatMults(metal_sum, buffer_lookup, runtime_items)
    left_node = transformer.visit(node.left)
    right_node = transformer.visit(node.right)
    # get the code for left operand, save it to cpu var
    # get the code for right operand, save it to cpu var
    # move each to tempA and tempB
    # call the kernel and output to out
    decls = []
    for N in ast.walk(left_node):
        if isinstance(N, ast.Name):
            if N.id not in decls:
                decls.append(N.id)
    buf_decls = "\n    ".join([
        f"device float* {arg} [[ buffer({buffer_lookup[arg]}) ]]," for arg in decls
    ])
    body = f"out[id] = {transpiler.get_expr(left_node)};"
    metal_matmult_setup = f"""
#include <metal_stdlib>
using namespace metal;

kernel void {func_name}(
    {buf_decls}
    device float* out [[ buffer(0) ]],
    uint id [[ thread_position_in_grid ]]
) {{
    {body}
}}
""".strip()
    with open("./metal_samples/metal_matmult_setup.metal", "w") as f:
        f.write(metal_matmult_setup)
    
    # Compile AFTER buffers are set up
    lib, error = device.newLibraryWithSource_options_error_(metal_matmult_setup, None, None)
    fn = lib.newFunctionWithName_(func_name)
    pipeline_state, _ = device.newComputePipelineStateWithFunction_error_(fn, None)

    # Setup and dispatch
    cmd_buf = cmd_queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()

    encoder.setComputePipelineState_(pipeline_state)
    for i, buf in enumerate(bufs):
        encoder.setBuffer_offset_atIndex_(buf, 0, i)

    # Run helper
    threads_per_group = Metal.MTLSizeMake(256, 1, 1)
    num_threadgroups = Metal.MTLSizeMake((n + 255) // 256, 1, 1)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_threadgroups, threads_per_group)

    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()
    count = left_node.inferred_type[1] * left_node.inferred_type[2]
    ptr = bufs[0].contents().as_buffer(4 * count)
    leftMat = np.frombuffer(ptr, dtype=np.float32).copy()       # Make a copy to avoid overwriting

    decls = []
    for N in ast.walk(right_node):
        if isinstance(N, ast.Name):
            if N.id not in decls:
                decls.append(N.id)
    buf_decls = "\n    ".join([
        f"device float* {arg} [[ buffer({buffer_lookup[arg]}) ]]," for arg in decls
    ])
    body = f"out[id] = {transpiler.get_expr(right_node)};"
    metal_matmult_setup = f"""
#include <metal_stdlib>
using namespace metal;

kernel void {func_name}(
    {buf_decls}
    device float* out [[ buffer(0) ]],
    uint id [[ thread_position_in_grid ]]
) {{
    {body}
}}
""".strip()
    with open("./metal_samples/metal_matmult_setup.metal", "w") as f:
        f.write(metal_matmult_setup)
    
    # Compile AFTER buffers are set up
    lib, error = device.newLibraryWithSource_options_error_(metal_matmult_setup, None, None)
    fn = lib.newFunctionWithName_(func_name)
    pipeline_state, _ = device.newComputePipelineStateWithFunction_error_(fn, None)

    # Setup and dispatch
    cmd_buf = cmd_queue.commandBuffer()
    encoder = cmd_buf.computeCommandEncoder()

    encoder.setComputePipelineState_(pipeline_state)
    for i, buf in enumerate(bufs):
        encoder.setBuffer_offset_atIndex_(buf, 0, i)

    # Run helper
    threads_per_group = Metal.MTLSizeMake(256, 1, 1)
    num_threadgroups = Metal.MTLSizeMake((n + 255) // 256, 1, 1)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_threadgroups, threads_per_group)

    encoder.endEncoding()
    cmd_buf.commit()
    cmd_buf.waitUntilCompleted()
    count = right_node.inferred_type[1] * right_node.inferred_type[2]
    ptr = bufs[0].contents().as_buffer(4 * count)
    rightMat = np.frombuffer(ptr, dtype=np.float32)

    overwrite_buffer(bufs[1], leftMat)
    overwrite_buffer(bufs[2], rightMat)

    dims = np.array([left_node.inferred_type[1], left_node.inferred_type[2], right_node.inferred_type[2]], dtype=np.uint32)

    descA = MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
        dims[0], dims[1], dims[1] * 4, MPSDataTypeFloat32)
    descB = MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
        dims[1], dims[2], dims[2] * 4, MPSDataTypeFloat32)
    descC = MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
        dims[0], dims[2], dims[2] * 4, MPSDataTypeFloat32)
    matA = MPSMatrix.alloc().initWithBuffer_descriptor_(bufs[1], descA)
    matB = MPSMatrix.alloc().initWithBuffer_descriptor_(bufs[2], descB)
    matC = MPSMatrix.alloc().initWithBuffer_descriptor_(bufs[0], descC)

    gemm = MPSMatrixMultiplication.alloc().initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
    device, False, False, dims[0], dims[2], dims[1], 1.0, 0.0)

    # Create new command buffer + encoder
    cmd_buf = cmd_queue.commandBuffer()
    gemm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(cmd_buf, matA, matB, matC)
    cmd_buf.commit()
    return "out"