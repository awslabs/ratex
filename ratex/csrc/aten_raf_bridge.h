/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"

namespace torch_lazy_tensors {
namespace bridge {
namespace raf_backend {

c10::optional<LazyTensor> TryGetLtcTensor(const at::Tensor& tensor);

// Extracts the LazyTensor out of our version of at::Tensor. Throws an exception
// if tensor is not a lazy tensor.
LazyTensor GetLtcTensor(const at::Tensor& tensor);

// Same as above, applied to a list of tensors.
std::vector<LazyTensor> GetLtcTensors(lazy_tensors::Span<const at::Tensor> tensors);

// Get client data from ir node output
lazy_tensors::ComputationClient::DataPtr GetData(ir::Output out);

// Create a new lazy tensor with the same metadata of the input tensor (with
// possible overrides), and the new IR value.
LazyTensor CreateFrom(const LazyTensor& self, ir::Value ir_value);

}  // namespace raf_backend
}  // namespace bridge
}  // namespace torch_lazy_tensors
