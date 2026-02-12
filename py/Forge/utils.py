def _default_strides(shape):
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _indexing_helper(self, key):
    if isinstance(key, list):
        raise TypeError(
            "Array: fancy indexing (passing lists/arrays as indices) is not supported"
        )
    if not isinstance(key, tuple):
        key = (key,)

    ellipsis_count = key.count(Ellipsis)
    if ellipsis_count > 1:
        raise IndexError("Array: only one ellipsis allowed in indexing")
    if ellipsis_count == 1:
        ellipsis_index = key.index(Ellipsis)
        explicit_count = sum(1 for k in key if k is not Ellipsis and k is not None)
        if len(self.shape) < explicit_count:
            raise IndexError("Array: too many indices for array")
        num_missing = len(self.shape) - explicit_count
        key = (
            key[:ellipsis_index]
            + (slice(None),) * num_missing
            + key[ellipsis_index + 1 :]
        )
        key = tuple(key)

    new_shape = list(self.shape)
    new_strides = list(self.strides)
    new_offset = self.offset

    dim = 0
    deleted = 0
    if len(self.shape) < len(key) - key.count(None):
        raise IndexError("Array: too many indices for array")
    for s in key:
        if s is None:
            new_shape.insert(dim, 1)
            new_strides.insert(dim, 0)
            deleted -= 1

        elif isinstance(s, int):
            if s < 0:
                s += self.shape[dim]
            if s < 0 or s >= self.shape[dim]:
                raise IndexError("Array: Index out of range")

            new_offset += s * new_strides[dim - deleted]
            new_shape.pop(dim - deleted)
            new_strides.pop(dim - deleted)
            deleted += 1
            dim += 1

        elif isinstance(s, slice):
            if s.step == 0:
                raise ValueError("Array: slice step cannot be zero")
            if s.step is None:
                step = 1
            else:
                step = s.step
            if s.start is None:
                if step > 0:
                    start = 0
                else:
                    start = self.shape[dim] - 1
            else:
                start = s.start
            if s.stop is None:
                if step > 0:
                    stop = self.shape[dim]
                elif self.shape[dim] == 0:
                    stop = 0
                else:
                    stop = -1 - self.shape[dim]
            else:
                stop = s.stop

            if start < 0:
                start += self.shape[dim]
            if stop < 0:
                stop += self.shape[dim]
            if start < 0:
                start = 0
            if start >= self.shape[dim] and self.shape[dim]:
                start = self.shape[dim] - 1
            if stop < -1:
                stop = -1
            if stop > self.shape[dim]:
                stop = self.shape[dim]

            new_offset += start * new_strides[dim - deleted]
            new_strides[dim - deleted] *= step
            sgn = 1 if step > 0 else -1
            new_shape[dim - deleted] = max(
                0, (sgn * (stop - start) + abs(step) - 1) // abs(step)
            )
            dim += 1

        else:
            raise TypeError("Array: Only int and slice supported")

    return new_shape, new_strides, new_offset
