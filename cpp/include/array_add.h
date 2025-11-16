#pragma once
#include <memory>
#include "array_handle.h"

std::shared_ptr<ArrayHandle> add_arrays_cpp(const std::shared_ptr<ArrayHandle>& A, 
    const std::shared_ptr<ArrayHandle>& B);
