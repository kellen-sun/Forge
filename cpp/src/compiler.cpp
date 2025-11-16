#include "../include/forge_handle.h"
#include "../include/compiler.h"
#include <string>

std::unique_ptr<ForgeHandle> compile_from_source_cpp(const std::string& src) {
    // 1. Parse Python source → AST (or your own DSL parsing)
    // 2. Build IR
    // 3. Generate Metal kernel source (or keep IR for later)
    return std::make_unique<ForgeHandle>("IR(" + src + ")");
}
