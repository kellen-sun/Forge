import pytest
from array import array as pyarray
from Forge import Array
import numpy as np

# --- BASIC SHAPE + CREATION ---

def test_shape_basic():
    a = Array([[1, 2, 3], [4, 5, 6]])
    assert a.shape == (2, 3)

def test_shape_1d():
    a = Array([1, 2, 3])
    assert a.shape == (3,)

def test_shape_scalar():
    a = Array(5.0)
    assert a.shape == ()
    assert a.list() == 5.0

def test_empty_array_top_level():
    arr = Array([])
    assert arr.shape == (0,)
    assert arr.list() == []

def test_empty_array_second_level():
    arr = Array([[]])
    assert arr.shape == (1, 0)
    assert arr.list() == [[]]

def test_wrong_shape_call():
    with pytest.raises(TypeError):
        Array([[1, 2], [1, 2]]).shape()

def test_empty_nested_list():
    with pytest.raises(ValueError):
        Array([[1, 2], []])

def test_ragged_2d():
    with pytest.raises(ValueError):
        Array([[1, 2], [3]])

def test_ragged_deep():
    with pytest.raises(ValueError):
        Array([[1, 2], [3, [4]]])


# --- TYPE HANDLING ---

def test_array_module_float_array():
    arr = pyarray("f", [1.0, 2.0, 3.0])
    a = Array(arr)
    assert a.shape == (3,)
    assert a.list() == [1.0, 2.0, 3.0]

def test_array_module_wrong_type():
    arr = pyarray("i", [1, 2, 3])
    with pytest.raises(TypeError):
        Array(arr)

def test_bytes_without_shape_fails():
    with pytest.raises(TypeError):
        Array(b'\x00\x01')

def test_bytes_with_shape():
    a = Array.from_buffer(np.array([1,2,3], dtype=np.float32), shape=(3,))
    assert a.shape == (3,)
    assert a.list() == [1.0, 2.0, 3.0]

def test_bytes_with_shape_fail():
    with pytest.raises(TypeError):
        Array(b'\x00\x00\x00\x01', shape=(2,3))


# --- ADDITION ---

def test_addition_correct():
    a1 = Array([[1.0, 2.0], [3.0, 4.0]])
    a2 = Array([[4.0, 5.0], [6.0, 7.0]])
    result = a1 + a2
    assert result.list() == [[5.0, 7.0], [9.0, 11.0]]

def test_addition_mismatch_shape():
    a1 = Array([[1, 2]])
    a2 = Array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        _ = a1 + a2

def test_radd():
    a1 = Array([1, 2, 3])
    a2 = Array([4, 5, 6])
    assert (a1 + a2).list() == [5, 7, 9]
    assert (a2 + a1).list() == [5, 7, 9]
    assert (a1.__radd__(a2)).list() == [5, 7, 9]

def test_add_non_array():
    a = Array([1, 2, 3])
    with pytest.raises(TypeError):
        _ = a + 5

# --- SUBTRACTION ---

def test_subtraction_correct():
    a1 = Array([[10.0, 5.0], [2.0, 8.0]])
    a2 = Array([[1.0, 2.0], [3.0, 4.0]])
    # Expected: [[9.0, 3.0], [-1.0, 4.0]]
    result = a1 - a2
    assert result.list() == [[9.0, 3.0], [-1.0, 4.0]]

def test_subtraction_mismatch_shape():
    a1 = Array([1, 2, 3])
    a2 = Array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        _ = a1 - a2

def test_subtraction_rsub():
    a1 = Array([10.0, 20.0])
    a2 = Array([5.0, 15.0])
    # a1 - a2 = [5.0, 5.0]
    assert (a1 - a2).list() == [5.0, 5.0]
    # a2 - a1 = [-5.0, -5.0]
    assert (a2 - a1).list() == [-5.0, -5.0]

def test_sub_non_array():
    a = Array([1, 2, 3])
    with pytest.raises(TypeError):
        _ = a - 5

# --- MULTIPLICATION ---

def test_multiplication_correct():
    a1 = Array([[2.0, 5.0], [4.0, 1.0]])
    a2 = Array([[3.0, 2.0], [0.5, 10.0]])
    # Expected: [[6.0, 10.0], [2.0, 10.0]]
    result = a1 * a2
    assert result.list() == [[6.0, 10.0], [2.0, 10.0]]

def test_multiplication_commutative():
    a1 = Array([1, 2])
    a2 = Array([3, 4])
    # Multiplication should be commutative (A * B == B * A)
    assert (a1 * a2).list() == (a2 * a1).list()
    assert (a1 * a2).list() == [3.0, 8.0]

def test_mult_non_array():
    a = Array([1, 2, 3])
    with pytest.raises(TypeError):
        _ = a * 5

# --- DIVISION ---

def test_division_correct():
    a1 = Array([[10.0, 8.0], [5.0, 1.0]])
    a2 = Array([[2.0, 4.0], [2.0, 2.0]])
    # Expected: [[5.0, 2.0], [2.5, 0.5]]
    result = a1 / a2
    assert result.list() == [[5.0, 2.0], [2.5, 0.5]]

def test_division_by_zero_safe_handling():
    # Test for standard IEEE 754 behavior (inf for X/0, NaN for 0/0)
    a1 = Array([1.0, 5.0, 0.0])
    a2 = Array([0.0, 1.0, 0.0])
    result = a1 / a2
    
    result_list = result.list()
    # 1.0 / 0.0 -> Infinity
    assert result_list[0] == float('inf') 
    # 5.0 / 1.0 -> 5.0
    assert result_list[1] == 5.0
    # 0.0 / 0.0 -> Not a Number (NaN)
    assert np.isnan(result_list[2])

def test_div_non_array():
    a = Array([1, 2, 3])
    with pytest.raises(TypeError):
        _ = a / 5

def test_div_mismatch_shape():
    a1 = Array([[1, 2]])
    a2 = Array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        _ = a1 / a2
    
# --- from_buffer + from_handle ---

def test_from_buffer():
    arr = pyarray("f", [1.0, 2.0, 3.0, 4.0])
    mv = memoryview(arr)
    a = Array.from_buffer(mv, shape=[2, 2])
    assert a.shape == (2, 2)
    assert a.list() == [[1.0, 2.0], [3.0, 4.0]]

def test_from_handle_roundtrip():
    a = Array([[1, 2], [3, 4]])
    h = a._handle
    b = Array.from_handle(h)
    assert b.shape == (2, 2)
    assert b.list() == [[1, 2], [3, 4]]


# --- DEEP NESTING ---

def test_3d_shape():
    a = Array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ])
    assert a.shape == (2, 2, 2)

def test_4d_shape():
    a = Array([ [[[1]], [[2]]], [[[3]], [[4]]] ])
    assert a.shape == (2, 2, 1, 1)


# --- EDGE CASES / SANITY ---

def test_zero_dim_addition():
    a = Array(3.0)
    b = Array(4.0)
    result = a + b
    assert result.list() == 7.0

def test_repr_contains_shape():
    a = Array([[1, 2], [3, 4]])
    assert "shape" in repr(a)

def test_list_roundtrip():
    original = [[1, 2], [3, 4]]
    a = Array(original)
    assert a.list() == original
