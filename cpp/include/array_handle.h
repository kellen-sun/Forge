#pragma once
#include "forge_handle.h"

#include <vector>
#include <span>
#include <cstdint>


class ArrayHandle {
private:
    std::vector<int64_t> shape_;
    void* metal_buffer_ = nullptr;
public:
    // CONSTRUCTORS //
    ArrayHandle(std::vector<int64_t> shape, void* dev = nullptr);
    ArrayHandle(const float* src_data, std::vector<int64_t> shape, void* dev = nullptr);
    ~ArrayHandle();

    // ACCESSORS //
    const std::vector<int64_t>& shape() const { return shape_; }
    std::span<const float> data() const;
    std::span<float> data();
    void* metal_buffer() const { return metal_buffer_; }

    // SETTER //
    void set_metal_buffer(void* buf) { metal_buffer_ = buf; }
};

inline std::shared_ptr<ForgeHandle> get_default_forge() {
    static std::once_flag once;
    static std::shared_ptr<ForgeHandle> inst;
    std::call_once(once, [&]() {
        inst = std::make_shared<ForgeHandle>();
    });
    return inst;
}

inline size_t numel_from_shape(const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (auto d : shape) n *= (size_t) d;
    return n;
}

std::vector<int64_t> array_shape(const std::shared_ptr<ArrayHandle>& h);
