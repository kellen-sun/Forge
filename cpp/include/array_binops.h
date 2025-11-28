#pragma once
#include <memory>
#include "array_handle.h"

enum class ArrayOperationType : int {
    ADD = 0,
    SUB = 1,
    MULT = 2,
    DIV = 3
};

std::shared_ptr<ArrayHandle> operations_arrays_cpp(const std::shared_ptr<ArrayHandle>& A, 
    const std::shared_ptr<ArrayHandle>& B,
    ArrayOperationType op_type);
    