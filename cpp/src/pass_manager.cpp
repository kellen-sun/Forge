#include "../include/pass.h"

void PassManager::addPass(std::unique_ptr<Pass> pass) { passes.push_back(std::move(pass)); }

void PassManager::run(IR& ir) {
    for (auto& pass : passes) {
        pass->run(ir);
    }
}
