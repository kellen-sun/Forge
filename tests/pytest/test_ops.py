import numpy as np
import pytest
from Forge import Array

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
    assert result_list[0] == float("inf")
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


# --- MATMUL ---


def test_matmul_2d_correct():
    a1 = Array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    a2 = Array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    result = a1 @ a2
    assert result.list() == [[58.0, 64.0], [139.0, 154.0]]


def test_matvec_correct():
    a1 = Array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    a2 = Array([7.0, 8.0, 9.0])
    result = a1 @ a2
    assert result.list() == [50.0, 122.0]


def test_vecmat_correct():
    a1 = Array([1.0, 2.0, 3.0])
    a2 = Array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    result = a1 @ a2
    assert result.list() == [58.0, 64.0]


def test_vecvec_correct():
    a1 = Array([1.0, 2.0, 3.0])
    a2 = Array([7.0, 9.0, 11.0])
    result = a1 @ a2
    assert result.list() == 58.0


def test_matmul_broadcast_shape():
    a1 = Array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    a2 = Array([[1.0, 2.0], [3.0, 4.0]])
    result = a1 @ a2  # Should broadcast a2 to shape (1, 2, 2)
    assert result.shape == (2, 2, 2)
    assert result.list() == [[[7.0, 10.0], [15.0, 22.0]], [[23.0, 34.0], [31.0, 46.0]]]


def test_matmul_mismatch_shape():
    a1 = Array([[1.0, 2.0], [3.0, 4.0]])
    a2 = Array([[5.0, 6.0, 7.0]])
    with pytest.raises(RuntimeError):
        _ = a1 @ a2


@pytest.fixture
def matrix_4x4():
    data = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    return Array(data)


@pytest.fixture
def identity_4x4():
    data = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    return Array(data)


def test_matmul_strided_rows(matrix_4x4, identity_4x4):
    a_strided = matrix_4x4[::2]
    res = a_strided @ identity_4x4

    assert res.shape == (2, 4)
    expected = [[0, 1, 2, 3], [8, 9, 10, 11]]
    assert res.list() == expected


def test_matmul_strided_cols(matrix_4x4):
    a_strided = matrix_4x4[:, ::2]
    b = Array([[1], [1]])
    res = a_strided @ b
    assert res.shape == (4, 1)
    expected = [[2], [10], [18], [26]]
    assert res.list() == expected


def test_matmul_double_trouble(matrix_4x4):
    a_strided = matrix_4x4[::2]
    b_data = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    b_full = Array(b_data)
    b_strided = b_full[:, ::2]
    res = a_strided @ b_strided
    expected = [[0, 2], [8, 10]]
    assert res.shape == (2, 2)
    assert res.list() == expected


def test_batch_broadcasting_strided():
    data = []
    for i in range(4):
        data.append([[i, i], [i, i]])

    a_full = Array(data)
    a_strided = a_full[1:3]

    b = Array([[1, 0], [0, 1]])

    res = a_strided @ b

    expected = [[[1, 1], [1, 1]], [[2, 2], [2, 2]]]
    assert res.shape == (2, 2, 2)
    assert res.list() == expected


def test_matmul_transpose():
    a1 = Array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).T
    a2 = Array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    result = a1 @ a2
    assert result.list() == [[89.0, 98.0], [116.0, 128.0]]
