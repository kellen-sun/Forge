#pragma once
#import <Metal/Metal.h>

#include <string>

id<MTLComputePipelineState> get_pipeline(const std::string& op_name, const char* metal_c_string);
