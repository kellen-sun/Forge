#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/buffer_info.h>

#include "forge_handle.h"

#include <vector>
#include <cstdint>

namespace py = pybind11;

class ArrayHandle {
private:
    // TODO: decide, eventually fade out data_ (?) - save mem, but can cache values?
    std::vector<float> data_;
    std::vector<int64_t> shape_;
    void* metal_buffer_ = nullptr;
public:
    // CONSTRUCTORS //
    ArrayHandle() = default;
    ArrayHandle(std::vector<float> data, std::vector<int64_t> shape, void* dev);
    ~ArrayHandle();

    // ACCESSORS //
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<float>& data() const { return data_; }
    std::vector<float>& data() { return data_; }
    // updates data from gpu buffer, instead of accessing vector data_
    const std::vector<float>& download() const;
    std::vector<float>& download();
    void* metal_buffer() const { return metal_buffer_; }

    // SETTER //
    void set_metal_buffer(void* buf) { metal_buffer_ = buf; }
};

static std::shared_ptr<ForgeHandle> get_default_forge() {
    static std::once_flag once;
    static std::shared_ptr<ForgeHandle> inst;
    std::call_once(once, [&]() {
        inst = std::make_shared<ForgeHandle>();
    });
    return inst;
}

std::vector<int64_t> array_shape(const std::shared_ptr<ArrayHandle>& h);

std::shared_ptr<ArrayHandle> create_array_from_buffer_py(py::buffer buf, std::vector<int64_t> shape,
    ForgeHandle* FH = nullptr);

py::object array_to_list(const std::shared_ptr<ArrayHandle>& h);