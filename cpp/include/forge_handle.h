#pragma once
#include <memory>
#include <string>

class ForgeHandle {
   private:
    std::string ir_repr_;
    // internal hidden implementation, pimpl pattern in .mm
    struct Impl;

   public:
    std::unique_ptr<Impl> impl;

    explicit ForgeHandle(const std::string& ir);
    ForgeHandle();
    ~ForgeHandle();
    ForgeHandle(const ForgeHandle&) = delete;
    ForgeHandle& operator=(const ForgeHandle&) = delete;
    ForgeHandle(ForgeHandle&&) noexcept;
    ForgeHandle& operator=(ForgeHandle&&) noexcept;

    const std::string& ir() const { return ir_repr_; }
    void* device_ptr() const;
    void* queue_ptr() const;
};
