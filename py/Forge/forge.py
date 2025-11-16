import inspect
from . import _backend
from .array import Array

class CompiledKernel:
    def __init__(self, handle):
        self.handle = handle
    
    def __call__(self, *args):
        handles = [a._handle for a in args]
        return Array.from_handle(_backend.run_kernel(self.handle, handles))

def forge(fn):
    """Decorator"""
    src = inspect.getsource(fn)

    handle = _backend.compile_from_source(src)
    return CompiledKernel(handle)

