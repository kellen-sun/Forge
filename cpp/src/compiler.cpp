#include "../include/compiler.h"

#include <string>

#include "../include/forge_handle.h"

// std::unique_ptr<ForgeHandle> compile_from_source_cpp(const std::string& src) {
//     auto handle = std::make_unique<ForgeHandle>();

//     @autoreleasepool {
//         NSError* err = nil;

//         id<MTLLibrary> lib =
//             [handle->impl->device newLibraryWithSource:
//                 [NSString stringWithUTF8String:src.c_str()]
//                 options:nil
//                 error:&err];

//         if (!lib) {
//             printf("Metal compile error: %s\n",
//                    [[err localizedDescription] UTF8String]);
//             return nullptr;
//         }

//         // For now assume there is 1 kernel named "main"
//         id<MTLFunction> fn = [lib newFunctionWithName:@"main"];
//         if (!fn) return nullptr;

//         NSError* perr = nil;
//         id<MTLComputePipelineState> p =
//             [handle->impl->device newComputePipelineStateWithFunction:fn
//                                                                 error:&perr];

//         PipelineInfo info;
//         info.pipeline = p;
//         info.threads = MTLSizeMake(32, 1, 1);

//         handle->impl->pipelines["main"] = info;
//     }

//     return handle;
// }
