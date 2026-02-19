import pytest
from Forge import Array


# Helper to create a shared 3D array for testing
# Shape: (2, 3, 4), Values: 0 to 23
@pytest.fixture
def tensor_3d():
    data = [
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
    ]
    return Array(data)


# --- transpose ---


def test_transpose(tensor_3d):
    view = tensor_3d.T
    assert view.shape == (4, 3, 2)
    assert view.list() == [
        [[0, 12], [4, 16], [8, 20]],
        [[1, 13], [5, 17], [9, 21]],
        [[2, 14], [6, 18], [10, 22]],
        [[3, 15], [7, 19], [11, 23]],
    ]

    view = tensor_3d.transpose(axes=[1, 0, 2])
    assert view.shape == (3, 2, 4)
    assert view.list() == [
        [[0, 1, 2, 3], [12, 13, 14, 15]],
        [[4, 5, 6, 7], [16, 17, 18, 19]],
        [[8, 9, 10, 11], [20, 21, 22, 23]],
    ]

    with pytest.raises(ValueError, match="axes don't match array"):
        _ = tensor_3d.transpose(axes=[2, 0])
    with pytest.raises(ValueError, match="axes must be a permutation"):
        _ = tensor_3d.transpose(axes=[0, 0, 1])


# --- reshape ---


def test_reshape(tensor_3d):
    view = tensor_3d.reshape((6, 4))
    assert view.shape == (6, 4)
    assert view.list() == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23],
    ]

    view = tensor_3d.reshape((2, -1))
    assert view.shape == (2, 12)
    assert view.list() == [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    ]

    with pytest.raises(ValueError, match="cannot reshape array of size"):
        _ = tensor_3d.reshape((5, 5))
    with pytest.raises(ValueError, match="reshape can only infer one dimension"):
        _ = tensor_3d.reshape((-1, -1))


def test_reshape_on_non_contiguous(tensor_3d):
    view_strided = tensor_3d[..., ::2]
    flat = view_strided.reshape(-1)

    assert flat.shape == (12,)
    expected = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    assert flat.list() == expected
    flat[0] = 999
    assert tensor_3d[0, 0, 0] == 0
    assert flat[0] == 999


def test_transpose_then_reshape(tensor_3d):
    transposed = tensor_3d.T
    flat = transposed.reshape(-1)

    assert flat.shape == (24,)
    assert flat[0] == 0
    assert flat[1] == 12

    flat[0] = 888
    assert tensor_3d[0, 0, 0] == 0


def test_mutation_propagation(tensor_3d):
    t = tensor_3d.T
    t[0, 0, 0] = 100
    assert tensor_3d[0, 0, 0] == 100

    tensor_3d[0, 0, 0] = 0
    r = tensor_3d.reshape((2, 12))
    r[0, 0] = 200
    assert tensor_3d[0, 0, 0] == 200
