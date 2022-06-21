/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ratex/csrc/raf_model_state.h"

namespace torch_lazy_tensors {

RAFModelState* RAFModelState::Get() {
  static RAFModelState model_state;
  return &model_state;
}

bool RAFModelState::IsModelState(const LazyTensor& tensor) {
  return model_state_.count(tensor.GetUniqueId());
}

void RAFModelState::AddModelState(const LazyTensor& tensor) {
  model_state_.insert(tensor.GetUniqueId());
}

bool RAFModelState::IsAMPEnabled() {
  return enable_amp_;
}

void RAFModelState::SetAMPEnabled(bool enabled) {
  enable_amp_ = enabled;
}

RAFModelState* GetRAFModelState() {
  return RAFModelState::Get();
}

}  // namespace torch_lazy_tensors
