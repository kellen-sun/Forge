#pragma once

#include <vector>

#include "../../cpp/include/graph.h"

inline Node make_input(std::vector<int64_t> shape, std::vector<int64_t> strides) {
    Node n;
    n.op = OpCode::INPUT;
    n.shape = std::move(shape);
    n.strides = std::move(strides);
    n.offset = 0;
    return n;
}

inline Node make_add(int a, int b, std::vector<int64_t> shape, std::vector<int64_t> strides) {
    Node n;
    n.op = OpCode::ADD;
    n.inputs = {a, b};
    n.shape = std::move(shape);
    n.strides = std::move(strides);
    n.offset = 0;
    return n;
}
