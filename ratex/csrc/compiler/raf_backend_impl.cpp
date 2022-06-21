/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ratex/csrc/compiler/base_backend_impl.h"
#include "client/raf_computation_client.h"

namespace torch_lazy_tensors {
namespace compiler {

class RAFBackendImpl : public BaseBackendImpl {
 public:
  lazy_tensors::ComputationClient* GetComputationClient() const override {
    // TODO: confirm extended RAF ComputationClient shall be in
    // pytorch-ltc/xla/lazy_xla/csrc/compiler/nnc_computation_client.h
    // or pytorch-ltc/xla/third_party/xla_client/computation_client.cc
    // return xla::compiler::NNCGet();
    return ratex::RAFGet();
  }

  lazy_tensors::ComputationClient* GetComputationClientIfInitialized() const override {
    // TODO: confirm extended RAF ComputationClient shall be in
    // pytorch-ltc/xla/lazy_xla/csrc/compiler/nnc_computation_client.h
    // or pytorch-ltc/xla/third_party/xla_client/computation_client.cc
    // return xla::compiler::NNCGetIfInitialized();
    return ratex::RAFGetIfInitialized();
  }
};

BackendImplRegistry* raf_backend_impl_registry =
    GetBackendImplRegistry()->AddBackendImpl(new RAFBackendImpl(), 10);

BackendRegistrar g_registrar(GetBackendImplRegistry()->GetBackendImpl());

}  // namespace compiler
}  // namespace torch_lazy_tensors
