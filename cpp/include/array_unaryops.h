#pragma once
#include <map>
#include <memory>

#include "array_handle.h"

std::shared_ptr<ArrayHandle> array_unaryops(const std::shared_ptr<ArrayHandle>& A,
                                            const std::string& op_name);
