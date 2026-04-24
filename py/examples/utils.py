# Used to unpack our binary data
import array
import struct

from Forge import Array


def load_data_to_forge(filename):
    with open(filename, "rb") as f:
        # Read the 12-byte header (3 integers * 4 bytes)
        header = f.read(12)
        num_items, rows, cols = struct.unpack("III", header)

        # Calculate total size
        img_count = num_items * rows * cols

        # Read images into a python array
        images = array.array("B")
        images.fromfile(f, img_count)

        # Read labels
        labels = array.array("B")
        labels.fromfile(f, num_items)

    # Convert to floats
    images_float = array.array("f", (float(x) / 255.0 for x in images))
    images = Array(images_float)
    one_hot_lab = (1.0 if i == label else 0.0 for label in labels for i in range(10))
    labels_float = array.array("f", one_hot_lab)
    labels = Array(labels_float)

    # Reshape
    images = images.reshape((num_items, -1))
    labels = labels.reshape((num_items, -1))

    return images, labels, (num_items, rows, cols)
