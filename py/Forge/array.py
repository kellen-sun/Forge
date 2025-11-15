from . import _backend
import numpy as np

class Array:
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.h = _backend.to_tensor(data)
        elif isinstance(data, _backend.TensorHandle):
            self.h = data
        else:
            raise TypeError("Unsupported type")

    def numpy(self):
        return _backend.from_tensor(self.h)

    def __add__(self, other):
        return Array(_backend.add(self.h, other.h))
