from Forge import Array
from Forge import _backend
print("backend has add_arrays:", hasattr(_backend, "add_arrays"))


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


a = Array([[1,2,3],[4,5,6]])
b = Array([[10,20,30],[40,50,60]])

c = a + b
print(c.list())
