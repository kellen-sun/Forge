import inspect
from . import _backend

class CompiledKernel:
    def __init__(self, handle):
        self.handle = handle
    
    def __call__(self, *args):
        return _backend.run_kernel(self.handle, list(args))

def forge(fn):
    """Decorator"""
    src = inspect.getsource(fn)

    def wrapper(*args):
        handle = _backend.compile_from_source(src)
        compiled = CompiledKernel(handle)
        return compiled(*args)

    wrapper._kernel_src = src
    return wrapper
