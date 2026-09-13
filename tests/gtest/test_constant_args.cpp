#include <gtest/gtest.h>

#include "../../cpp/include/compiler.h"

TEST(ConstantArgs, DecodeMatchesPythonStructPack) {
    // 3.5 == 0x40600000
    EXPECT_FLOAT_EQ(decode_f32(0x40600000), 3.5f);
    EXPECT_EQ(encode_f32(3.5f), 0x40600000);
    EXPECT_FLOAT_EQ(decode_f32(encode_f32(1.5f)), 1.5f);
}
