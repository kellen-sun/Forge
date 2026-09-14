#pragma once

#include <memory>
#include <vector>

#include "ir.h"

void verify(const IR& ir);

// a pass mutates the IR in place.
class Pass {
   public:
    virtual ~Pass() = default;
    virtual const char* name() const = 0;
    virtual void run(IR& ir) = 0;
};

class PassManager {
   public:
    void addPass(std::unique_ptr<Pass> pass);
    void run(IR& ir);

   private:
    std::vector<std::unique_ptr<Pass>> passes;
};

class DCEPass : public Pass {
   public:
    const char* name() const override { return "dce"; }
    void run(IR& ir) override;
};
