#pragma once

#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"

namespace torch_lazy_tensors {
namespace bridge {
namespace mnm_backend {

c10::optional<LazyTensor> TryGetLtcTensor(const at::Tensor& tensor);

// Extracts the LazyTensor out of our version of at::Tensor. Throws an exception
// if tensor is not a lazy tensor.
LazyTensor GetLtcTensor(const at::Tensor& tensor);

// Same as above, applied to a list of tensors.
std::vector<LazyTensor> GetLtcTensors(lazy_tensors::Span<const at::Tensor> tensors);

}  // namespace mnm_backend
}  // namespace bridge
}  // namespace torch_lazy_tensors
