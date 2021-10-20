#include "torch_mnm/csrc/aten_mnm_bridge.h"
#include "torch_mnm/csrc/mnm_model_state.h"

#include "client/base_computation_client.h"

#include "lazy_tensor_core/csrc/ops/device_data.h"

namespace torch_lazy_tensors {
namespace bridge {
namespace mnm_backend {

using namespace ir;

lazy_tensors::ComputationClient::DataPtr GetData(Output out) {
  auto* device_data = ir::ops::DeviceData::Cast(out.node);
  if (device_data) {
    return device_data->data();
  }
  for (size_t i = 0; i < out.node->operands().size(); ++i) {
    auto data = GetData(out.node->operand(i));
    if (data) {
      return data;
    }
  }
  return nullptr;
}

c10::optional<LazyTensor> TryGetLtcTensor(const at::Tensor& tensor) {
  c10::optional<LazyTensor> lazy_tensor = ::torch_lazy_tensors::bridge::TryGetLtcTensor(tensor);
  if (lazy_tensor && GetMNMModelState()->IsModelState(*lazy_tensor)) {
    Value value = lazy_tensor->GetIrValue();   
    lazy_tensors::ComputationClient::DataPtr data = GetData(value);
    static_cast<torch_mnm::BaseComputationClient::BaseData*>(data.get())->is_param = true;
  }
  return lazy_tensor;
}

LazyTensor GetLtcTensor(const at::Tensor& tensor) {
  LazyTensor lazy_tensor = ::torch_lazy_tensors::bridge::GetLtcTensor(tensor);
  if (GetMNMModelState()->IsModelState(lazy_tensor)) {
    Value value = lazy_tensor.GetIrValue();   
    lazy_tensors::ComputationClient::DataPtr data = GetData(value);
    static_cast<torch_mnm::BaseComputationClient::BaseData*>(data.get())->is_param = true;
  }
  return lazy_tensor; 
}

std::vector<LazyTensor> GetLtcTensors(
    lazy_tensors::Span<const at::Tensor> tensors) {
  std::vector<LazyTensor> ltc_tensors;
  ltc_tensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    ltc_tensors.push_back(bridge::GetLtcTensor(tensor));
  }
  return ltc_tensors;
}

}  // namespace mnm_backend
}  // namespace bridge
}  // namespace torch_lazy_tensors
