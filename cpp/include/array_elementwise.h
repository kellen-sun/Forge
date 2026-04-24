#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "array_handle.h"

std::shared_ptr<ArrayHandle> array_nullaryops(const std::vector<int64_t>& shape,
                                              const std::string& op_name);

std::shared_ptr<ArrayHandle> array_unaryops(const std::shared_ptr<ArrayHandle>& A,
                                            const std::string& op_name);

std::shared_ptr<ArrayHandle> array_binops(const std::shared_ptr<ArrayHandle>& A,
                                          const std::shared_ptr<ArrayHandle>& B,
                                          const std::string& op_name);

std::shared_ptr<ArrayHandle> array_inplaceops(const std::shared_ptr<ArrayHandle>& A,
                                              const std::shared_ptr<ArrayHandle>& B,
                                              const std::string& op_name);
