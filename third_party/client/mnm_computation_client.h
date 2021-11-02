#pragma once
#include "client/base_computation_client.h"
#include "lazy_tensors/computation_client/computation_client.h"
#include "lazy_tensors/computation_client/client_data.h"
#include "mnm/value.h"
#include "mnm/ir.h"

namespace torch_mnm {

using namespace lazy_tensors;

class MNMComputationClient : public BaseComputationClient {
 public:
  struct MNMData : public BaseData {
    MNMData(std::string device, Shape shape, bool is_param = false)
        : BaseData(std::move(device), GetShapeData(std::move(shape)), is_param) {
    }
    MNMData(std::string device, Shape shape, mnm::value::Value handle, bool is_param = false)
        : BaseData(std::move(device), GetShapeData(std::move(shape)), is_param), handle(handle) {
    }

    int64 get_handle() const {
      return reinterpret_cast<int64>(handle.get());
    }

    OpaqueHandle GetOpaqueHandle() override {
      return get_handle();
    }

    void Assign(const Data& data) override;

    bool HasValue() const override {
      return handle.defined();
    }

    /*! \brief TupleValue or TensorValue */
    mnm::value::Value handle;
  };

  struct MNMComputation : public BaseComputation {
    MNMComputation(std::shared_ptr<GenericComputation> computation, ProgramShape program_shape,
                   std::vector<std::string> devices,
                   const std::unordered_map<int64_t, int64_t>& alias = {})
        : BaseComputation(computation, program_shape, devices, alias) {
    }

    MNMComputation(std::shared_ptr<GenericComputation> computation, ProgramShape program_shape,
                   std::vector<std::string> devices, tvm::runtime::Module executable,
                   tvm::runtime::Module vm_module,
                   const std::unordered_map<int64_t, int64_t>& alias = {})
        : BaseComputation(computation, program_shape, devices, alias),
          executable(executable),
          vm_module(vm_module) {
    }

    tvm::runtime::Module executable;
    tvm::runtime::Module vm_module;
  };

  MNMComputationClient(Options options);

  static std::unique_ptr<ComputationClient> Create();

  DataPtr CreateDataPlaceholder(std::string device, Shape shape) override;

  std::vector<DataPtr> TransferToServer(lazy_tensors::Span<const TensorSource> tensors) override;

  std::vector<Literal> TransferFromServer(lazy_tensors::Span<const DataPtr> handles) override;

  std::vector<ComputationPtr> Compile(std::vector<CompileInstance> instances) override;

  std::vector<DataPtr> ExecuteComputation(const Computation& computation,
                                          lazy_tensors::Span<const DataPtr> arguments,
                                          const std::string& device,
                                          const ExecuteComputationOptions& options) override;

 private:
  std::vector<DataPtr> TransferToServerInternal(lazy_tensors::Span<const TensorSource> tensors);
};

lazy_tensors::ComputationClient* MNMGet();

lazy_tensors::ComputationClient* MNMGetIfInitialized();

}  // namespace torch_mnm
