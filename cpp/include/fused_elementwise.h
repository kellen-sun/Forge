#pragma once

#include <cstddef>
#include <cstdint>

enum class FusedOp : int64_t {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    UNARY = 4,
};

inline constexpr int64_t kFusedEncodingVersion = 1;
inline constexpr size_t kFusedHeaderSize = 3;
inline constexpr size_t kFusedInstructionWidth = 4;

inline constexpr int64_t fused_temp_ref(int64_t index) {
    return -1 - index;
}
