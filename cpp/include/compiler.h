#pragma once
#include "forge_handle.h"

std::unique_ptr<ForgeHandle> compile_from_source_cpp(const std::string& src);
