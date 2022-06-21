/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "client/base_computation_client.h"
#include "lazy_tensors/computation_client/computation_client.h"
#include "lazy_tensors/computation_client/client_data.h"
#include "raf/value.h"
#include "raf/ir.h"

namespace ratex {

using namespace lazy_tensors;

class RAFComputationClient : public BaseComputationClient {
 public:
  struct RAFData : public BaseData {
    RAFData(std::string device, Shape shape, bool is_param = false)
        : BaseData(std::move(device), GetShapeData(std::move(shape)), is_param) {
    }
    RAFData(std::string device, Shape shape, raf::value::Value handle, bool is_param = false)
        : BaseData(std::move(device), GetShapeData(std::move(shape)), is_param), handle(handle) {
    }

    int64_t get_handle() const {
      return reinterpret_cast<int64_t>(handle.get());
    }

    OpaqueHandle GetOpaqueHandle() override {
      return get_handle();
    }

    void Assign(const Data& data) override;

    bool HasValue() const override {
      return handle.defined();
    }

    /*! \brief TupleValue or TensorValue */
    raf::value::Value handle;
  };

  struct RAFComputation : public BaseComputation {
    RAFComputation(std::shared_ptr<GenericComputation> computation, ProgramShape program_shape,
                   std::vector<std::string> devices,
                   const std::unordered_map<int64_t, int64_t>& alias = {})
        : BaseComputation(computation, program_shape, devices, alias) {
    }

    RAFComputation(std::shared_ptr<GenericComputation> computation, ProgramShape program_shape,
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

  RAFComputationClient(Options options);

  static std::unique_ptr<ComputationClient> Create();

  DataPtr CreateDataPlaceholder(std::string device, Shape shape) override;

  std::vector<DataPtr> TransferToServer(lazy_tensors::Span<const TensorSource> tensors) override;

  std::vector<Literal> TransferFromServer(lazy_tensors::Span<const DataPtr> handles) override;

  ComputationPtr Compile(CompileInstance instances) override;

  std::vector<DataPtr> ExecuteComputation(const Computation& computation,
                                          lazy_tensors::Span<const DataPtr> arguments,
                                          const std::string& device,
                                          const ExecuteComputationOptions& options) override;

  std::vector<DataPtr> DryrunComputation(const Computation& computation,
                                         lazy_tensors::Span<const DataPtr> arguments,
                                         const std::string& device,
                                         const ExecuteComputationOptions& options);

 private:
  std::vector<DataPtr> TransferToServerInternal(lazy_tensors::Span<const TensorSource> tensors);
};

lazy_tensors::ComputationClient* RAFGet();

lazy_tensors::ComputationClient* RAFGetIfInitialized();

}  // namespace ratex
