import pytest
from Forge import Array

# Helper to create a shared 3D array for testing
# Shape: (2, 3, 4), Values: 0 to 23
@pytest.fixture
def tensor_3d():
    data = [
        [
            [0, 1, 2, 3], 
            [4, 5, 6, 7], 
            [8, 9, 10, 11]
        ],
        [
            [12, 13, 14, 15], 
            [16, 17, 18, 19], 
            [20, 21, 22, 23]
        ]
    ]
    return Array(data)

# --- __getitem__ ---

def test_basic_integer_indexing(tensor_3d):
    view = tensor_3d[1]
    assert view.shape == (3, 4)
    assert view.list() == [
        [12, 13, 14, 15], 
        [16, 17, 18, 19], 
        [20, 21, 22, 23]
    ]

    view = tensor_3d[0, 1]
    assert view.shape == (4,)
    assert view.list() == [4, 5, 6, 7]

def test_scalar_return(tensor_3d):
    """Test that fully indexing down to a single element returns a float (scalar), not an Array."""
    val = tensor_3d[0, 2, 3]
    assert isinstance(val, float)
    assert val == 11.0

def test_negative_indexing(tensor_3d):
    """Test negative indices."""
    assert tensor_3d[-1].list() == tensor_3d[1].list()
    assert tensor_3d[0, -1].list() == [8, 9, 10, 11]

def test_basic_slicing(tensor_3d):
    """Test standard range slicing."""
    view = tensor_3d[0, 1:2]
    assert view.shape == (1, 4)
    assert view.list() == [[4, 5, 6, 7]]

    view = tensor_3d[0:1]
    assert view.shape == (1, 3, 4)
    assert view.list() == [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]

def test_strided_slicing(tensor_3d):
    """Test slicing with steps."""
    view = tensor_3d[0, :, ::2]
    assert view.shape == (3, 2)
    assert view.list() == [[0, 2], [4, 6], [8, 10]]

def test_errors_out_of_bounds(tensor_3d):
    with pytest.raises(IndexError, match="Index out of range"):
        _ = tensor_3d[5]

    with pytest.raises(IndexError, match="Index out of range"):
        _ = tensor_3d[0, 5]

def test_negative_step(tensor_3d):
    """
    Test for step < 0 and stop < start.
    """
    view = tensor_3d[::-1]
    assert view.shape == (2, 3, 4)

    view = tensor_3d[0, 2:0:-1]
    assert view.shape == (2, 4)
    assert view.list() == [[8, 9, 10, 11], [4, 5, 6, 7]]

    view = tensor_3d[0, 1, 3:0:-2]
    assert view.shape == (2,)
    assert view.list() == [7, 5]

    view = tensor_3d[2:1]
    assert view.shape == (0, 3, 4)

    view = tensor_3d[1][2:-2:-1]
    assert view.shape == (1,4)
    assert view.list() == [[20, 21, 22, 23]]

def test_extreme_range_step(tensor_3d):
    """
    Way out of range.
    """
    view = tensor_3d[-13:40:2]
    assert view.shape == (1, 3, 4)
    assert view.list() == [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]

    view = tensor_3d[-12:40:2]
    assert view.shape == (1, 3, 4)
    assert view.list() == [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]

def test_zero_step(tensor_3d):
    """
    Test for step = 0.
    """
    with pytest.raises(ValueError, match="slice step cannot be zero"):
        _ = tensor_3d[::0]

def test_errors_too_many_indices(tensor_3d):
    with pytest.raises(IndexError, match="too many indices"):
        _ = tensor_3d[0, 0, 0, 0]

def test_chain_indexing_vs_tuple(tensor_3d):
    """Verify that a[x][y] behaves same as a[x, y]"""
    tuple_view = tensor_3d[0, 1]
    chain_view = tensor_3d[0][1]
    
    assert tuple_view.list() == chain_view.list()
    assert tuple_view.shape == chain_view.shape

# --- None & Ellipsis ---

def test_ellipsis_indexing(tensor_3d):
    """Test ... expansion."""
    view = tensor_3d[..., 0]
    assert view.shape == (2, 3)
    assert view.list() == [[0, 4, 8], [12, 16, 20]]

    assert tensor_3d[0, ...].list() == tensor_3d[0].list()

def test_newaxis_expansion(tensor_3d):
    """Test inserting new dimensions with None."""
    # Shape is (2, 3, 4). a[None] -> (1, 2, 3, 4)
    view = tensor_3d[None]
    assert view.shape == (1, 2, 3, 4)
    assert view.list() == [tensor_3d.list()]

    view = tensor_3d[:, None, :]
    assert view.shape == (2, 1, 3, 4)

# --- __setitem__ ---

def test_set_indexing(tensor_3d):
    tensor_3d[1,2] = [30, 31, 32, 33]
    assert tensor_3d.shape == (2, 3, 4)
    assert tensor_3d.list()[1][2] == [30, 31, 32, 33]

def test_set_broadcasting(tensor_3d):
    tensor_3d[1,1] = 3
    assert tensor_3d.shape == (2, 3, 4)
    assert tensor_3d.list()[1][1] == [3, 3, 3, 3]

    tensor_3d[1,0] = [7]
    assert tensor_3d.shape == (2, 3, 4)
    assert tensor_3d[1,0].list() == [7, 7, 7, 7]

# --- sum & len ---

def test_sum(tensor_3d):
    total = sum(tensor_3d)
    assert total.shape == (3,4)
    assert total.list() == [
        [12.0, 14.0, 16.0, 18.0],
        [20.0, 22.0, 24.0, 26.0],
        [28.0, 30.0, 32.0, 34.0]
    ]
    tot2 = sum(total)
    assert tot2.shape == (4,)
    assert tot2.list() == [60.0, 66.0, 72.0, 78.0]
    final = sum(tot2)
    assert isinstance(final, float)
    assert final == 276.0

def test_len(tensor_3d):
    assert len(tensor_3d) == 2
    assert len(tensor_3d[0]) == 3
    assert len(tensor_3d[0,0]) == 4