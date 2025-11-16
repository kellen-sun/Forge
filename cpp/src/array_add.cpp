#include "../include/array_add.h"

std::shared_ptr<ArrayHandle> add_arrays_cpp(const std::shared_ptr<ArrayHandle>& A, 
  const std::shared_ptr<ArrayHandle>& B) {
    const auto& shapeA = A->shape();
    const auto& shapeB = B->shape();

    if (shapeA != shapeB) {
        throw std::runtime_error("add_arrays_cpp: shape mismatch");
    }

    const auto& dataA = A->data();
    const auto& dataB = B->data();

    std::vector<float> out(dataA.size());
    for (size_t i = 0; i < dataA.size(); ++i)
        out[i] = dataA[i] + dataB[i];

    return std::make_shared<ArrayHandle>(std::move(out), shapeA);
}
