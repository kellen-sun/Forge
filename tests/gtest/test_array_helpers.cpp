#include <gtest/gtest.h>

#include <stdexcept>

#include "../../cpp/include/array_handle.h"

TEST(ArrayHelpersTest, numel) {
    std::vector<int64_t> shape1{5, 6, 2, 9};
    ASSERT_EQ(numel_from_shape(shape1), 540);

    std::vector<int64_t> shape2{0, 6, 2, 9};
    ASSERT_EQ(numel_from_shape(shape2), 0);

    std::vector<int64_t> shape3{1, 1, 1, 2};
    ASSERT_EQ(numel_from_shape(shape3), 2);
}

TEST(ArrayHelpersTest, make_strides) {
    std::vector<int64_t> shape1{5, 6, 2, 9};
    std::vector<int64_t> strides1{108, 18, 9, 1};
    ASSERT_EQ(make_strides(shape1), strides1);

    std::vector<int64_t> shape2{0, 6, 2, 9};
    std::vector<int64_t> strides2{108, 18, 9, 1};
    ASSERT_EQ(make_strides(shape2), strides2);

    std::vector<int64_t> shape3{1, 1, 1, 2};
    std::vector<int64_t> strides3{2, 2, 2, 1};
    ASSERT_EQ(make_strides(shape3), strides3);
}

TEST(ArrayHelpersTest, matmul_output_shape) {
    EXPECT_EQ(matmul_output_shape({2, 3}, {3, 4}), (std::vector<int64_t>{2, 4}));
    EXPECT_EQ(matmul_output_shape({3}, {3}), (std::vector<int64_t>{}));
    EXPECT_EQ(matmul_output_shape({3}, {3, 4}), (std::vector<int64_t>{4}));
    EXPECT_EQ(matmul_output_shape({2, 3}, {3}), (std::vector<int64_t>{2}));
    EXPECT_EQ(matmul_output_shape({5, 2, 3}, {3, 4}), (std::vector<int64_t>{5, 2, 4}));
    EXPECT_THROW(matmul_output_shape({2, 3}, {4, 5}), std::runtime_error);
}
