from Forge import Array
import pytest

def test():
    """name"""
    a = Array([[1, 2, 3], [4,5,6]])
    assert a.shape == (2,3)

def test_addition():
    a1 = Array([[1.0, 2.0], [3.0, 4.0]])
    a2 = Array([[4.0, 5.0], [6.0, 7.0]])
    result = a1 + a2

    assert result.list() == [[5.0, 7.0], [9.0, 11.0]]

def test_wrong_size():
    with pytest.raises(ValueError):
        Array([[1, 2, 3], []])

def test_empty_array():
    arr = Array([])
    assert arr.shape == (0,)
    assert arr.list() == []