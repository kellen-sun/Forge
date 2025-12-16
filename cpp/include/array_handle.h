#pragma once
#include <cstdint>
#include <span>
#include <vector>

#include "forge_handle.h"

struct ArrayStorage {
    void* metal_buffer_ = nullptr;
    void* write_event_ = nullptr;

    ~ArrayStorage();
};
class ArrayHandle {
   private:
    std::vector<int64_t> shape_;
    std::vector<int64_t> strides_;
    size_t offset_;
    std::shared_ptr<ArrayStorage> storage_;

   public:
    // CONSTRUCTORS //
    ArrayHandle(std::vector<int64_t> shape, void* dev = nullptr);
    ArrayHandle(const float* src_data, std::vector<int64_t> shape, void* dev = nullptr);
    ArrayHandle(const std::shared_ptr<ArrayHandle>& parent, std::vector<int64_t> new_shape,
                std::vector<int64_t> new_strides, size_t new_offset);
    ~ArrayHandle();

    // ACCESSORS //
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& strides() const { return strides_; }
    size_t offset() const { return offset_; }
    std::span<const float> data() const;
    std::span<float> data();
    void* metal_buffer() const;

    // SETTER //
    void set_metal_buffer(void* buf);
    void set_event(void* event);
    void copy_from(std::shared_ptr<ArrayHandle> other, std::vector<int64_t> shape,
                   std::vector<int64_t> strides, size_t offset);

    void synchronize();
};

inline std::shared_ptr<ForgeHandle> get_default_forge() {
    static std::once_flag once;
    static std::shared_ptr<ForgeHandle> inst;
    std::call_once(once, [&]() { inst = std::make_shared<ForgeHandle>(); });
    return inst;
}

inline size_t numel_from_shape(const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (auto d : shape) n *= (size_t)d;
    return n;
}

std::vector<int64_t> array_shape(const std::shared_ptr<ArrayHandle>& h);

std::shared_ptr<ArrayHandle> array_reshape(const std::shared_ptr<ArrayHandle>& h,
                                           std::vector<int64_t> shape);
