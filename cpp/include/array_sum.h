#pragma once
#include <map>
#include <memory>

#include "array_handle.h"

std::shared_ptr<ArrayHandle> sum_global(const std::shared_ptr<ArrayHandle>& A, bool keepdims);

std::shared_ptr<ArrayHandle> sum_axis(const std::shared_ptr<ArrayHandle>& A, size_t axis,
                                      bool keepdims);
