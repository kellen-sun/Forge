from forge import metal
import numpy as np

@metal
def mult(a, b):
    return a*sum(sum(b)*b)

a = np.array([1.0, 2.0, 6.0], dtype=np.float32)
b = np.array([3.0, 4.0, 7.0], dtype=np.float32)
print(mult(a, b))
