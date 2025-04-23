import Metal
import numpy as np
from Foundation import NSURL

# Create a Metal device
device = Metal.MTLCreateSystemDefaultDevice()
url = NSURL.fileURLWithPath_("add_arrays.metallib")
lib, error = device.newLibraryWithURL_error_(url, None)

if lib is None:
    print("Failed to load Metal library:", error)
else:
    print("Library loaded!")


# Get kernel function
fn = lib.newFunctionWithName_("add_arrays")

# Create pipeline
pipeline_state, _ = device.newComputePipelineStateWithFunction_error_(fn, None)

# Prepare data
n = 2048*1024*4
a_np = np.random.rand(n).astype(np.float32)
b_np = np.random.rand(n).astype(np.float32)
print("Input A[0:5]:", a_np[:5])
print("Input B[0:5]:", b_np[:5])
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
print("Result[0:5]:", result[:5])
