#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../include/array_handle.h"
#include "../include/forge_handle.h"
#include "../include/runtime.h"

struct ForgeHandle::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;

    Impl() {
        device = MTLCreateSystemDefaultDevice();
        queue = [device newCommandQueue];
    }
};

// TODO: Delete this constructor?
ForgeHandle::ForgeHandle(const std::string& ir) : ir_repr_(std::move(ir)) {}

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
