from forge import *
import numpy as np
import time
@forge
def matmul(a, b):
    return a @ b

A = np.random.rand(4096, 4096).astype(np.float32)
B = np.random.rand(4096, 4096).astype(np.float32)
x, y = [], []
for i in range(10):
    t = time.time()
    matmul(A, B)
    x.append(time.time() - t)
    t = time.time()
    A @ B
    y.append(time.time() - t)
print("Metal: ", sum(x)/len(x))
print("NumPy: ", sum(y)/len(y))
print(matmul(A, B) == A @ B)