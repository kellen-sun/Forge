import pytest
from Forge import Array

# Helper to create a predictable 3D array for testing
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

def test_basic_integer_indexing(tensor_3d):
    """Test reducing dimensions via integer indexing."""
    # a[1] -> Selects second block
    view = tensor_3d[1]
    assert view.shape == (3, 4)
    assert view.list() == [
        [12, 13, 14, 15], 
        [16, 17, 18, 19], 
        [20, 21, 22, 23]
    ]

    # a[0, 1] -> Selects first block, second row
    view = tensor_3d[0, 1]
    assert view.shape == (4,)
    assert view.list() == [4, 5, 6, 7]


def test_scalar_return(tensor_3d):
    """Test that fully indexing down to a single element returns a float (scalar), not an Array."""
    # a[0, 2, 3] -> Value 11
    val = tensor_3d[0, 2, 3]
    assert isinstance(val, float)
    assert val == 11.0

def test_negative_indexing(tensor_3d):
    """Test wrapping negative indices."""
    # a[-1] should be same as a[1]
    assert tensor_3d[-1].list() == tensor_3d[1].list()
    
    # a[0, -1] should be same as a[0, 2] (last row of first block)
    assert tensor_3d[0, -1].list() == [8, 9, 10, 11]

def test_basic_slicing(tensor_3d):
    """Test standard range slicing."""
    # Slice middle row: a[0, 1:2, :]
    view = tensor_3d[0, 1:2]
    assert view.shape == (1, 4)
    assert view.list() == [[4, 5, 6, 7]]

    # Slice range: a[0:1]
    view = tensor_3d[0:1]
    assert view.shape == (1, 3, 4)
    assert view.list() == [[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]]

def test_strided_slicing(tensor_3d):
    """Test slicing with steps (strides)."""
    # a[0, :, ::2] -> First block, all rows, every 2nd column
    # Rows are [0,1,2,3] -> [0, 2]
    view = tensor_3d[0, :, ::2]
    assert view.shape == (3, 2)
    assert view.list() == [[0, 2], [4, 6], [8, 10]]

def test_errors_out_of_bounds(tensor_3d):
    """Test IndexError logic."""
    with pytest.raises(IndexError, match="Index out of range"):
        _ = tensor_3d[5]
        
    with pytest.raises(IndexError, match="Index out of range"):
        _ = tensor_3d[0, 5]

def test_errors_invalid_slice(tensor_3d):
    """
    Test the specific slice restrictions in your implementation.
    Your code explicitly raises IndexError for step <= 0 and stop < start.
    """
    with pytest.raises(IndexError, match="slice step must be positive"):
        _ = tensor_3d[::-1]

    # Your implementation forbids stop < start (unlike standard python lists which return empty)
    with pytest.raises(IndexError, match="slice stop less than start"):
        _ = tensor_3d[2:1]

def test_errors_too_many_indices(tensor_3d):
    with pytest.raises(IndexError, match="too many indices"):
        _ = tensor_3d[0, 0, 0, 0]