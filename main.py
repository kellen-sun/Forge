from forge import *
import numpy as np

@metal
def matmul(a, b):
    return a @ b

A = np.random.rand(4096, 4096).astype(np.float32)
B = np.random.rand(4096, 4096).astype(np.float32)

matmul(A, B)
