import inspect
import ast
import Metal
import numpy as np
from transpiler import transpile, get_args, get_func_args

class TypeAnnotator(ast.NodeVisitor):
    def __init__(self, var_types):
        self.var_types = var_types
        self.builtins = {
            'sum': 'scalar',
            'len': 'scalar'
        }

    def visit_Name(self, node):
        node.inferred_type = self.var_types.get(node.id, 'unknown')

    def visit_BinOp(self, node):
        self.visit(node.left)
        self.visit(node.right)
        ltype = getattr(node.left, 'inferred_type', 'unknown')
        rtype = getattr(node.right, 'inferred_type', 'unknown')
        if ltype == rtype:
            node.inferred_type = ltype
        elif 'array' in (ltype[0], rtype[0]):
            node.inferred_type = ltype if ltype[0] == 'array' else rtype
        elif ltype[0] == 'matrix' and rtype[0] == 'matrix' and type(node.op) == ast.MatMult and ltype[2] == rtype[1]:
            node.inferred_type = ('matrix', ltype[1], rtype[2])
        else:
            node.inferred_type = 'scalar'
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in self.builtins:
                node.inferred_type = self.builtins[func_name]
            else:
                node.inferred_type = 'unknown'
        self.generic_visit(node)


def infer_arg_types(func_args, args):
    inferred = {}
    for name, val in zip(func_args, args):
        if isinstance(val, (int, float)):
            inferred[name] = "scalar"
        elif isinstance(val, np.ndarray):
            if val.ndim == 1:
                inferred[name] = ("array", val.shape[0])
            elif val.ndim == 2:
                inferred[name] = ("matrix", val.shape[0], val.shape[1])
            else:
                inferred[name] = f"unsupported_dim_{val.ndim}"
        else:
            inferred[name] = "unknown"
    return inferred

def metal(func):
    source = inspect.getsource(func)
    tree = ast.parse(source)
    
    metal_args = get_args(tree)
    func_args = get_func_args(tree)

    def wrapper(*args, **kwargs):
        arg_types = infer_arg_types(func_args, args)
        TypeAnnotator(arg_types).visit(tree)

        output_type = tree.body[0].body[0].value.inferred_type
        if output_type == 'scalar':
            s = 1
        elif output_type[0] == 'array':
            s = output_type[1]
        elif output_type[0] == 'matrix':
            s = output_type[1] * output_type[2]

        device = Metal.MTLCreateSystemDefaultDevice()

        # what size should temp bufferes and in-body variables be?
        np_inputs = []
        # Running assumption that function args are arrays or matrices
        for i in range(len(func_args)):
            np_inputs.append(args[i].ravel())
        for i in range(len(metal_args)):
            if metal_args[i] not in func_args:
                np_inputs.append(np.empty_like(args[0]))

        # Allocate Metal buffers
        bufs = [
            device.newBufferWithLength_options_(np_inputs[0].size * 4, Metal.MTLResourceStorageModeShared),
            device.newBufferWithLength_options_(args[0].size * 4, Metal.MTLResourceStorageModeShared),
            device.newBufferWithLength_options_(args[0].size * 4, Metal.MTLResourceStorageModeShared)
        ]
        dims = []
        for arg in func_args:
            # Running assumption that function args are arrays or matrices
            dims = dims + list(arg_types[arg][1:])
        bufs.append(device.newBufferWithBytes_length_options_(np.array(dims, dtype=np.uint32), len(dims) * 4, Metal.MTLResourceStorageModeShared))
        for i in range(len(func_args)):
            mv = memoryview(np_inputs[i])
            bufs.append(device.newBufferWithBytes_length_options_(mv, mv.nbytes, Metal.MTLResourceStorageModeShared))
        for i in range(len(metal_args)):
            if metal_args[i] not in func_args:
                bufs.append(device.newBufferWithLength_options_(np_inputs[i].size * 4, Metal.MTLResourceStorageModeShared))

        cmd_queue = device.newCommandQueue()
        source_str = transpile(tree, source, (device, bufs, s, cmd_queue))
        if not isinstance(source_str, str):
            return source_str
        options = None
        lib, error = device.newLibraryWithSource_options_error_(source_str, options, None)
        if lib is None:
            print("Failed to load Metal library:", error)
        fn = lib.newFunctionWithName_(func.__name__)

        # Create pipeline
        pipeline_state, _ = device.newComputePipelineStateWithFunction_error_(fn, None)

        # Create command buffer + encoder
        cmd_buf = cmd_queue.commandBuffer()
        encoder = cmd_buf.computeCommandEncoder()

        # Set up encoder
        encoder.setComputePipelineState_(pipeline_state)
        for i, buf in enumerate(bufs):
            encoder.setBuffer_offset_atIndex_(buf, 0, i)

        # Launch thread groups
        if output_type == 'scalar':
            threads_per_group = Metal.MTLSizeMake(32, 1, 1)
            num_threadgroups = Metal.MTLSizeMake((s + 31) // 32, 1, 1)
        elif output_type[0] == 'array':
            threads_per_group = Metal.MTLSizeMake(256, 1, 1)
            num_threadgroups = Metal.MTLSizeMake((s + 255) // 256, 1, 1)
        elif output_type[0] == 'matrix':
            threads_per_group = Metal.MTLSizeMake(16, 16, 1)
            num_threadgroups = Metal.MTLSizeMake((output_type[1]+15)//16, (output_type[2]+15)//16, 1)
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(num_threadgroups, threads_per_group)

        # Finalize and commit
        encoder.endEncoding()
        cmd_buf.commit()
        cmd_buf.waitUntilCompleted()

        # Access the raw memory from the Metal buffer
        ptr = bufs[0].contents().as_buffer(4 * s)
        result = np.frombuffer(ptr, dtype=np.float32)
        if output_type == 'scalar':
            result = result[0]
        elif output_type[0] == 'array':
            result = result.reshape(output_type[1])
        elif output_type[0] == 'matrix':
            result = result.reshape(output_type[1], output_type[2])
        else:
            raise ValueError(f"Unsupported output type: {output_type}")
        return result

    return wrapper

