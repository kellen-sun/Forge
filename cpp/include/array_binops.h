#pragma once
#include <map>
#include <memory>

#include "array_handle.h"

std::vector<int64_t> broadcast_shapes(std::span<const int64_t>& a_shape,
                                      std::span<const int64_t>& b_shape);

std::shared_ptr<ArrayHandle> array_binops(const std::shared_ptr<ArrayHandle>& A,
                                          const std::shared_ptr<ArrayHandle>& B,
                                          const std::string& op_name);

std::pair<std::shared_ptr<ArrayHandle>, bool> prepare(const std::shared_ptr<ArrayHandle>& h);

std::shared_ptr<ArrayHandle> array_matmul(const std::shared_ptr<ArrayHandle>& A,
                                          const std::shared_ptr<ArrayHandle>& B);
