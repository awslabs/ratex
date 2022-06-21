/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "lazy_tensors/computation_client/computation_client.h"
#include "lazy_tensors/computation_client/client_data.h"
#include "raf/value.h"
#include "raf/ir.h"

namespace ratex {

using namespace lazy_tensors;

class BaseComputationClient : public lazy_tensors::ComputationClient {
  struct DeviceHandle {
    std::string device;
    int64_t handle;
  };

  using Data = lazy_tensors::client::Data;

 public:
  struct BaseData : public Data {
    BaseData(std::string device, Shape shape, bool is_param = false)
        : Data(std::move(device), GetShapeData(std::move(shape))), is_param(is_param) {
    }

    /*! \brief Whether it is a Parameter or not */
    bool is_param{false};
  };

  struct BaseComputation : public Computation {
    BaseComputation(std::shared_ptr<GenericComputation> computation, ProgramShape program_shape,
                    std::vector<std::string> devices,
                    const std::unordered_map<int64_t, int64_t>& alias = {})
        : Computation(computation, program_shape, devices), alias(alias) {
    }

    std::unordered_map<int64_t, int64_t> alias;
  };

  struct Device {
    Device() = default;
    Device(const std::string& device_str);

    std::string kind;
    int ordinal = 0;
  };

  struct Worker {
    Worker(std::string name, int task_no) : name(std::move(name)), task_no(task_no) {
    }

    bool operator<(const Worker& rhs) const {
      if (task_no != rhs.task_no) {
        return task_no < rhs.task_no;
      }
      return name.compare(rhs.name) < 0;
    }

    bool operator==(const Worker& rhs) const {
      return task_no == rhs.task_no && name == rhs.name;
    }

    std::string name;
    int task_no;
  };

  struct Options {
    std::string default_device;
    /*! \brief Maps a PyTorch device ID (example, "GPU:0", "TPU:0") to the full
     *  coordinates in self device format.
     */
    std::map<std::string, std::string> global_device_map;
    /*! \brief These are the devices that this instance of PyTorch is handling. These
     *  devices are in the form of "CPU:0", "TPU:3", ... For each of these
     *  devices, there is an entry within the global_device_map.
     */
    std::set<std::string> devices;
    // Maps a TPU Worker with an EndPoint.
    // std::map<Worker, std::string> workers_map;
    /*! \brief Whether to enable persistent cache. Note that if enabled, the following
     *  methods must be impelmented:
     *    1) virtual std::string CompileSerialize(ComputationPtr instance);
     *    2) virtual ComputationPtr CompileDeSerialize(const std::string& str);
     *  Cache miss is expected when one of the following changes:
     *    1) Relay IR
     *    2) Shape
     *    3) The set of trianable parameters
     *    4) Input/output alias
     */
    bool cache_enabled{false};
  };

  BaseComputationClient(Options options) : options_(options) {
  }

  std::vector<std::vector<DataPtr>> ExecuteReplicated(
      const Computation& computation, const std::vector<std::vector<DataPtr>>& arguments,
      lazy_tensors::Span<const std::string> devices,
      const ExecuteReplicatedOptions& options) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<std::vector<DataPtr>> ExecuteParallel(
      lazy_tensors::Span<const Computation* const> computations,
      const std::vector<std::vector<DataPtr>>& arguments,
      lazy_tensors::Span<const std::string> devices,
      const ExecuteParallelOptions& options) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<DataPtr> ExecuteChained(lazy_tensors::Span<const ExecuteChainedOp> ops,
                                      const std::string& device) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<std::vector<DataPtr>> DeconstructTuple(
      lazy_tensors::Span<const DataPtr> tuples) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  std::string GetResourceDomain(const std::string& device) const override;

  std::string GetDefaultDevice() const override;

  size_t GetNumDevices() const override {
    return 1;
  }

  std::vector<std::string> GetLocalDevices() const override;

  std::vector<std::string> GetAllDevices() const override;

  void SetReplicationDevices(std::shared_ptr<std::vector<std::string>> devices) override;

  std::shared_ptr<std::vector<std::string>> GetReplicationDevices() override;

  void SetRngSeed(size_t seed) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  std::map<std::string, lazy_tensors::Metric> GetMetrics() const override {
    return {};
  }

  MemoryInfo GetMemoryInfo(const std::string& device) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  void PrepareToExit() override;

  std::vector<ComputationPtr> Compile(std::vector<CompileInstance> instances) override;

  virtual ComputationPtr Compile(CompileInstance instance) = 0;

  virtual raf::ObjectRef CompileCacheKey(CompileInstance instance);

  virtual std::string CompileSerialize(ComputationPtr instance) {
    LTC_LOG(FATAL) << "Serialization not implemented. Cached compilation should be disabled";
  }

  virtual ComputationPtr CompileDeSerialize(const std::string& json_path) {
    LTC_LOG(FATAL) << "DeSerialization not implemented. Cached compilation should be disabled";
  }

  virtual void SaveArtifacts(const std::string& dir, const std::string& json);

 protected:
  static lazy_tensors::client::ShapeData GetShapeData(const Shape& shape);

 protected:
  std::unordered_map<const Computation*, tvm::IRModule> lifted_computation_;

 private:
  Options options_;

  void DumpComputationAlias(const CompileInstance& instance, std::string path);
};

void PopulateLocalDevices(BaseComputationClient::Options* options);

}  // namespace ratex
