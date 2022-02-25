/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "razor/csrc/mnm_model_state.h"

namespace torch_lazy_tensors {

MNMModelState* MNMModelState::Get() {
  static MNMModelState model_state;
  return &model_state;
}

bool MNMModelState::IsModelState(const LazyTensor& tensor) {
  return model_state_.count(tensor.GetUniqueId());
}

void MNMModelState::AddModelState(const LazyTensor& tensor) {
  model_state_.insert(tensor.GetUniqueId());
}

bool MNMModelState::IsAMPEnabled() {
  return enable_amp_;
}

void MNMModelState::SetAMPEnabled(bool enabled) {
  enable_amp_ = enabled;
}

MNMModelState* GetMNMModelState() {
  return MNMModelState::Get();
}

}  // namespace torch_lazy_tensors
