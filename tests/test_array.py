from Forge import Array

# nested list -> create backend array
a = Array([[1,2,3],[4,5,6]])
print("shape:", a.shape)
print("as list:", a.list())

# also test array('f') path
from array import array
buf = array('f', [1.0, 2.0, 3.0, 4.0])
b = Array(buf)
print("shape buf:", b.shape)
print("b list:", b.list())
