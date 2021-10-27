#include "torch_mnm/csrc/compiler/base_backend_impl.h"
#include "client/mnm_computation_client.h"

namespace torch_lazy_tensors {
namespace compiler {

class MNMBackendImpl : public BaseBackendImpl {
 public:
  lazy_tensors::ComputationClient* GetComputationClient() const override {
    // TODO: confirm extended MNM ComputationClient shall be in
    // pytorch-ltc/xla/lazy_xla/csrc/compiler/nnc_computation_client.h
    // or pytorch-ltc/xla/third_party/xla_client/computation_client.cc
    // return xla::compiler::NNCGet();
    return torch_mnm::MNMGet();
  }

  lazy_tensors::ComputationClient* GetComputationClientIfInitialized() const override {
    // TODO: confirm extended MNM ComputationClient shall be in
    // pytorch-ltc/xla/lazy_xla/csrc/compiler/nnc_computation_client.h
    // or pytorch-ltc/xla/third_party/xla_client/computation_client.cc
    // return xla::compiler::NNCGetIfInitialized();
    return torch_mnm::MNMGetIfInitialized();
  }
};

BackendImplRegistry* mnm_backend_impl_registry =
    GetBackendImplRegistry()->AddBackendImpl(new MNMBackendImpl(), 10);

BackendRegistrar g_registrar(GetBackendImplRegistry()->GetBackendImpl());

}  // namespace compiler
}  // namespace torch_lazy_tensors
