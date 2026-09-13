#pragma once
#include <cstdint>
#include <cstring>

#include "graph.h"
#include "ir.h"

// Python _flatten packs CONSTANT as bytes
inline int64_t encode_f32(float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return static_cast<int64_t>(bits);
}

inline float decode_f32(int64_t arg) {
    uint32_t bits = static_cast<uint32_t>(arg);
    float v = 0.0f;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

void optimize_graph(IR& ir);

void generateKernels(Graph& graph);

void compile_metal(Graph& graph);
