#pragma once
#include <map>
#include <memory>

#include "array_handle.h"

std::shared_ptr<ArrayHandle> array_nullaryops(const std::vector<int64_t>& shape,
                                              const std::string& op_name);
