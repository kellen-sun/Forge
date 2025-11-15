#pragma once
#include <string>
#include <memory>

class ForgeHandle {
private:
    std::string ir_repr_;
public:
    ForgeHandle(std::string ir) : ir_repr_(std::move(ir)) {}
    
    const std::string& ir() const { return ir_repr_; }
};
