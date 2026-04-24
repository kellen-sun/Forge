from . import ops, shape
from .array import Array
from .forge import forge
from .utils import _set_seed

# package version
__version__ = "0.0.1"

for op_name in ops.UNARY_OPS:
    globals()[op_name] = getattr(ops, op_name)

for op_name in ops.NULLARY_OPS:
    globals()[op_name] = getattr(ops, op_name)

globals()["set_seed"] = _set_seed

__all__ = [
    "forge",
    "Array",
    "ops",
    "shape",
] + ops.UNARY_OPS
