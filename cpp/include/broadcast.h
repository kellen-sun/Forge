#pragma once
#include <cstdint>
#include <span>
#include <vector>

std::vector<int64_t> broadcast_shapes(std::span<const int64_t> a_shape,
                                      std::span<const int64_t> b_shape);

std::vector<int64_t> get_bcast_strides(std::span<const int64_t> shape,
                                       std::span<const int64_t> strides,
                                       std::span<const int64_t> final_shape);
