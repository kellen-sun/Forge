from . import ops, shape
from .array import Array
from .forge import forge

# package version
__version__ = "0.0.1"

for op_name in ops.UNARY_OPS:
    globals()[op_name] = getattr(ops, op_name)

__all__ = [
    "forge",
    "Array",
    "ops",
    "shape",
] + ops.UNARY_OPS
