#include <stdexcept>
#include <string>

#include "../include/pass.h"

void PassManager::addPass(std::unique_ptr<Pass> pass) { passes.push_back(std::move(pass)); }

void PassManager::run(IR& ir) {
    verify(ir);
    for (auto& pass : passes) {
        pass->run(ir);
        try {
            verify(ir);
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string(pass->name()) + ": " + e.what());
        }
    }
}
