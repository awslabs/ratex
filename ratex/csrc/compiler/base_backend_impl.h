/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "ratex/csrc/compiler/raf_lowering_context.h"
#include "ratex/csrc/compiler/raf_node_lowering.h"
#include "ratex/csrc/compiler/backend_registry.h"

namespace torch_lazy_tensors {
namespace compiler {

class BaseBackendImpl : public BackendImplInterface {
 public:
  std::unique_ptr<NodeLowering> CreateNodeLowering(ir::LoweringContext* loctx) const override {
    return CreateRAFNodeLowering(loctx);
  }

  NodeLowering* GetNodeLowering() const override {
    return GetRAFNodeLowering();
  }

  std::unique_ptr<ir::LoweringContext> CreateLoweringContext(
      const std::string& name, Device device, lazy_tensors::Span<const ir::Node* const> post_order,
      ir::Util::EmissionMap emit_status) const override {
    return std::make_unique<raf_backend::RAFLoweringContext>(name, device, post_order, emit_status);
  }

  std::unique_ptr<ir::LoweringContext> CreateLoweringContext(const std::string& name,
                                                             Device device) const override {
    return std::make_unique<raf_backend::RAFLoweringContext>(name, device);
  }

  std::vector<std::string> GetCompilationDevices(
      const std::string& device, lazy_tensors::Span<const std::string> devices) const override {
    return std::vector<std::string>(devices.begin(), devices.end());
  }

  at::Tensor MakeTensorFromComputationData(
      const lazy_tensors::ComputationClient::DataPtr data,
      c10::optional<at::ScalarType> logical_scalar_type) const override {
    auto ltc_literals = lazy_tensors::ComputationClient::Get()->TransferFromServer({data});
    LTC_CHECK_EQ(ltc_literals.size(), 1);
    LTC_CHECK(logical_scalar_type);
    return torch_lazy_tensors::MakeTensorFromLiteral(ltc_literals.front(), *logical_scalar_type);
  }

  lazy_tensors::ComputationClient::DataPtr MakeComputationDataFromTensor(
      const at::Tensor& tensor, const lazy_tensors::Shape& shape,
      const std::string& device) const override {
    std::vector<lazy_tensors::ComputationClient::TensorSource> source_tensors;
    Device physical_device(device);
    auto populate_fn = [&, device](
                           const lazy_tensors::ComputationClient::TensorSource& source_tensor,
                           void* dest_buffer, size_t dest_buffer_size) {
      PopulateTensorBuffer(tensor, source_tensor.shape, dest_buffer, dest_buffer_size,
                           physical_device);
    };
    source_tensors.emplace_back(lazy_tensors::ToShapeData(shape), device, std::move(populate_fn));
    auto handles = lazy_tensors::ComputationClient::Get()->TransferToServer(source_tensors);
    LTC_CHECK_EQ(handles.size(), 1);
    return handles.front();
  }

  lazy_tensors::StatusOr<std::string> GetComputationBackendText(
      const lazy_tensors::GenericComputation* computation) const override {
    LTC_LOG(FATAL) << "Not implemented.";
  }
};

}  // namespace compiler
}  // namespace torch_lazy_tensors
