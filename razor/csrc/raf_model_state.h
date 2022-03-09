/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <unordered_set>

#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"
#include "lazy_tensor_core/csrc/tensor.h"

namespace torch_lazy_tensors {

class RAFModelState {
 public:
  static RAFModelState* Get();

  bool IsModelState(const LazyTensor& tensor);
  void AddModelState(const LazyTensor& tensor);

  bool IsAMPEnabled();
  void SetAMPEnabled(bool enabled);

 private:
  bool enable_amp_ = false;
  std::unordered_set<int64_t> model_state_;
};

RAFModelState* GetRAFModelState();

}  // namespace torch_lazy_tensors
