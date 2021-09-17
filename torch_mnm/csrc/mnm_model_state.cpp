#include "torch_mnm/csrc/mnm_model_state.h"

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

MNMModelState* GetMNMModelState() { return MNMModelState::Get(); }

}  // namespace torch_lazy_tensors
