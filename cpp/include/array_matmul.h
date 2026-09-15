#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "array_handle.h"

struct MatmulPlan {
    bool a_vector;
    bool b_vector;
    int64_t m;
    int64_t k;
    int64_t n;
    int64_t a_row_stride;
    int64_t a_col_stride;
    int64_t b_row_stride;
    int64_t b_col_stride;
    std::vector<int64_t> batch_shape;
    std::vector<int64_t> a_batch_strides;
    std::vector<int64_t> b_batch_strides;
    std::vector<int64_t> output_shape;
};

MatmulPlan make_matmul_plan(const std::vector<int64_t>& a_shape,
                            const std::vector<int64_t>& a_strides,
                            const std::vector<int64_t>& b_shape,
                            const std::vector<int64_t>& b_strides);

std::shared_ptr<ArrayHandle> array_matmul(const std::shared_ptr<ArrayHandle>& A,
                                          const std::shared_ptr<ArrayHandle>& B);
