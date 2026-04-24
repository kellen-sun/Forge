#import <Metal/Metal.h>

#include "../include/array_handle.h"
#include "../include/forge_handle.h"

struct ForgeHandle::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    uint32_t seed;

    Impl() {
        device = MTLCreateSystemDefaultDevice();
        queue = [device newCommandQueue];
        seed = 42;
    }
};

ForgeHandle::ForgeHandle() : impl(std::make_unique<Impl>()) {
    @autoreleasepool {
        impl->device = MTLCreateSystemDefaultDevice();
        impl->queue = [impl->device newCommandQueue];
    }
}

ForgeHandle::~ForgeHandle() = default;
ForgeHandle::ForgeHandle(ForgeHandle&&) noexcept = default;
ForgeHandle& ForgeHandle::operator=(ForgeHandle&&) noexcept = default;

void* ForgeHandle::device_ptr() const { return (__bridge void*)impl->device; }

void* ForgeHandle::queue_ptr() const { return (__bridge void*)impl->queue; }

uint32_t ForgeHandle::get_seed() const { return impl->seed; }

void ForgeHandle::set_seed(uint32_t s) { impl->seed = s; }
