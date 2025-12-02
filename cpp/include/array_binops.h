#pragma once
#include <memory>
#include <map>
#include "array_handle.h"

std::shared_ptr<ArrayHandle> binops_arrays_cpp(const std::shared_ptr<ArrayHandle>& A, 
    const std::shared_ptr<ArrayHandle>& B,
    const std::string& op_name);
    