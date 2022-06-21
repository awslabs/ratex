/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ratex/csrc/aten_raf_bridge.h"
#include "ratex/csrc/raf_model_state.h"

#include "client/base_computation_client.h"

#include "lazy_tensor_core/csrc/ops/cast.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "lazy_tensor_core/csrc/tensor_util.h"

namespace torch_lazy_tensors {
namespace bridge {
namespace raf_backend {

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
  if (lazy_tensor && GetRAFModelState()->IsModelState(*lazy_tensor)) {
    Value value = lazy_tensor->GetIrValue();
    lazy_tensors::ComputationClient::DataPtr data = GetData(value);
    static_cast<ratex::BaseComputationClient::BaseData*>(data.get())->is_param = true;
  }
  return lazy_tensor;
}

LazyTensor GetLtcTensor(const at::Tensor& tensor) {
  LazyTensor lazy_tensor = ::torch_lazy_tensors::bridge::GetLtcTensor(tensor);
  if (GetRAFModelState()->IsModelState(lazy_tensor)) {
    Value value = lazy_tensor.GetIrValue();
    lazy_tensors::ComputationClient::DataPtr data = GetData(value);
    static_cast<ratex::BaseComputationClient::BaseData*>(data.get())->is_param = true;
  }
  return lazy_tensor;
}

std::vector<LazyTensor> GetLtcTensors(lazy_tensors::Span<const at::Tensor> tensors) {
  std::vector<LazyTensor> ltc_tensors;
  ltc_tensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    ltc_tensors.push_back(bridge::GetLtcTensor(tensor));
  }
  return ltc_tensors;
}

// TODO: Remove this function after we have control over LTC
ir::Value MaybeCastIrValue(const LazyTensor& self, ir::Value ir_value, const Device& device,
                           c10::optional<at::ScalarType> logical_element_type) {
  if (!logical_element_type) {
    logical_element_type = self.dtype_optional();
  }
  if (logical_element_type && RequiresRawTypeCasting(*logical_element_type, &device)) {
    ir_value = ir::MakeNode<ir::ops::Cast>(ir_value, *logical_element_type);
  }
  return ir_value;
}

// TODO: Remove this function after we have control over LTC
LazyTensor CreateFrom(const LazyTensor& self, ir::Value ir_value) {
  ir_value = MaybeCastIrValue(self, std::move(ir_value), self.GetDevice(),
                              /*logical_element_type=*/c10::nullopt);
  return LazyTensor::Create(std::move(ir_value), self.GetDevice(), self.dtype_optional());
}

}  // namespace raf_backend
}  // namespace bridge
}  // namespace torch_lazy_tensors
