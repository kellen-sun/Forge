from decorator import metal
import numpy as np

@metal
def add_arrays(a, b):
    return a + b

a = np.array([1.0, 2.0], dtype=np.float32)
b = np.array([3.0, 4.0], dtype=np.float32)
print(add_arrays(a, b))  # Should print [4.0, 6.0]
