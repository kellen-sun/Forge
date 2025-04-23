from forge import *
import numpy as np

@metal
def avg(a, b):
    return sum(a * b)

a = np.array([1.0, 2.0, 6.0], dtype=np.float32)
b = np.array([3.0, 4.0, 7.0], dtype=np.float32)
print(avg(a, b))
