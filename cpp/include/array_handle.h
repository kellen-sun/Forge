#pragma once
#include <vector>
#include <cstdint>

class ArrayHandle {
private:
    std::vector<float> data_;
    std::vector<int64_t> shape_;
public:
    // CONSTRUCTORS //
    ArrayHandle() = default;
    ArrayHandle(std::vector<float>&& data, std::vector<int64_t>&& shape)
     : data_{std::move(data)}, shape_{std::move(shape)} {}
    
    // ACCESSORS //
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<float>& data() const { return data_; }
    std::vector<float>& data() { return data_; }
};
