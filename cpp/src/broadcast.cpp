#include "../include/broadcast.h"

#include <algorithm>
#include <stdexcept>

std::vector<int64_t> broadcast_shapes(std::span<const int64_t> a_shape,
                                      std::span<const int64_t> b_shape) {
    std::vector<int64_t> out;
    auto it_a = a_shape.rbegin();
    auto it_b = b_shape.rbegin();
    auto end_a = a_shape.rend();
    auto end_b = b_shape.rend();

    while (it_a != end_a || it_b != end_b) {
        int64_t dim_a = (it_a != end_a) ? *it_a : 1;
        int64_t dim_b = (it_b != end_b) ? *it_b : 1;

        if (dim_a == dim_b) {
            out.push_back(dim_a);
        } else if (dim_a == 1) {
            out.push_back(dim_b);
        } else if (dim_b == 1) {
            out.push_back(dim_a);
        } else {
            throw std::runtime_error("broadcast_shapes: shapes cannot be broadcast");
        }

        if (it_a != end_a) ++it_a;
        if (it_b != end_b) ++it_b;
    }
    std::reverse(out.begin(), out.end());
    return out;
}

std::vector<int64_t> get_bcast_strides(std::span<const int64_t> shape,
                                       std::span<const int64_t> strides,
                                       std::span<const int64_t> final_shape) {
    std::vector<int64_t> bcast_strides(final_shape.size(), 0);
    int64_t offset = (int64_t)final_shape.size() - (int64_t)shape.size();

    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 1 && final_shape[offset + i] > 1) {
            bcast_strides[offset + i] = 0;
        } else {
            bcast_strides[offset + i] = strides[i];
        }
    }
    return bcast_strides;
}
