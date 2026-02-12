#pragma once
#include <memory>
#include <string>

class ForgeHandle {
   private:
    struct Impl;

   public:
    std::unique_ptr<Impl> impl;

    ForgeHandle();
    ~ForgeHandle();
    ForgeHandle(const ForgeHandle&) = delete;
    ForgeHandle& operator=(const ForgeHandle&) = delete;
    ForgeHandle(ForgeHandle&&) noexcept;
    ForgeHandle& operator=(ForgeHandle&&) noexcept;

    void* device_ptr() const;
    void* queue_ptr() const;
};
