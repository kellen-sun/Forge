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
