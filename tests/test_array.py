from Forge import Array

def test():
    """name"""
    a = Array([[1, 2, 3], [4,5,6]])
    assert a.shape == (2,3)