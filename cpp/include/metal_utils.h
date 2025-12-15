#pragma once
#include <string>
#import <Metal/Metal.h>

id<MTLComputePipelineState> get_pipeline(const std::string& op_name, const char* metal_c_string);