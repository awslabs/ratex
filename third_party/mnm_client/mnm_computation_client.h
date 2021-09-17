#include "lazy_tensors/computation_client/computation_client.h"
#include "lazy_tensors/computation_client/client_data.h"
#include "mnm/value.h"
#include "mnm/ir.h"

namespace torch_mnm {

using namespace lazy_tensors;

class MNMComputationClient : public lazy_tensors::ComputationClient {
  struct DeviceHandle {
    std::string device;
    lazy_tensors::int64 handle;
  };

  using Data = lazy_tensors::client::Data;

 public:
  struct MNMData : public Data {
    MNMData(std::string device, Shape shape)
        : Data(std::move(device), GetShapeData(std::move(shape))) {}
    // MNMData(std::string device, Shape shape,
    //         mnm::value::Value handle)
    //     : Data(std::move(device), GetShapeData(std::move(shape))),
    //       handle(handle) { }
    MNMData(std::string device, Shape shape,
            mnm::value::Value handle, bool is_param = false)
        : Data(std::move(device), GetShapeData(std::move(shape))),
          handle(handle), is_param(is_param) { }

    int64 get_handle() const { return reinterpret_cast<int64>(handle.get()); }

    OpaqueHandle GetOpaqueHandle() override { return get_handle(); }

    void Assign(const Data& data) override;

    bool HasValue() const override { return handle.defined(); }

    /*! \brief TupleValue or TensorValue */
    mnm::value::Value handle;
    /*! \brief Whether it is a Parameter or not */
    bool is_param{false};
  };

  struct MNMComputation : public Computation {
    MNMComputation(std::shared_ptr<GenericComputation> computation,
                   ProgramShape program_shape, std::vector<std::string> devices,
                   tvm::runtime::Module executable)
        : Computation(computation, program_shape, devices), executable(executable) {}

    MNMComputation(std::shared_ptr<GenericComputation> computation,
                   ProgramShape program_shape, std::vector<std::string> devices,
                   int neff_cnt, const std::unordered_map<int64_t, int64_t>& alias = {})
        : Computation(computation, program_shape, devices), neff_cnt(neff_cnt), alias(alias) {}

    tvm::runtime::Module executable;
    int neff_cnt{-1};
    std::unordered_map<int64_t, int64_t> alias;
  };

  struct Device {
    Device() = default;
    Device(const std::string& device_str);

    std::string kind;
    int ordinal = 0;
  };

  struct Worker {
    Worker(std::string name, int task_no)
        : name(std::move(name)), task_no(task_no) {}

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
    // Maps a PyTorch device ID (example, "GPU:0", "TPU:0") to the full
    // coordinates in TF device format
    // (ie, /job:tpu_worker/replica:0/task:0/device:TPU:0), of the worker
    // exposing that device. These devices are all the devices present within
    // the TPU mesh.
    std::map<std::string, std::string> global_device_map;
    // These are the devices that this instance of PyTorch is handling. These
    // devices are in the form of "CPU:0", "TPU:3", ... For each of these
    // devices, there is an entry within the global_device_map.
    std::set<std::string> devices;
    // Maps a TPU Worker with an EndPoint.
    // std::map<Worker, std::string> workers_map;
  };

  MNMComputationClient(Options options);

  DataPtr CreateDataPlaceholder(std::string device, Shape shape) override;

  std::vector<DataPtr> TransferToServer(
      lazy_tensors::Span<const TensorSource> tensors) override;

  std::vector<Literal> TransferFromServer(
      lazy_tensors::Span<const DataPtr> handles) override;

  std::vector<ComputationPtr> Compile(
      std::vector<CompileInstance> instances) override;

  std::vector<ComputationPtr> CompileSunda(
      std::vector<CompileInstance> instances);

  std::vector<DataPtr> ExecuteComputation(
      const Computation& computation, lazy_tensors::Span<const DataPtr> arguments,
      const std::string& device,
      const ExecuteComputationOptions& options) override;

  std::vector<DataPtr> ExecuteSundaComputation(
      const Computation& computation, lazy_tensors::Span<const DataPtr> arguments,
      const std::string& device,
      const ExecuteComputationOptions& options);

  std::vector<std::vector<DataPtr>> ExecuteReplicated(
      const Computation& computation,
      const std::vector<std::vector<DataPtr>>& arguments,
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

  std::vector<DataPtr> ExecuteChained(
      lazy_tensors::Span<const ExecuteChainedOp> ops,
      const std::string& device) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  std::vector<std::vector<DataPtr>> DeconstructTuple(
      lazy_tensors::Span<const DataPtr> tuples) override {
    LTC_LOG(FATAL) << "Not implemented yet.";
  }

  std::string GetResourceDomain(const std::string& device) const override;

  std::string GetDefaultDevice() const override;

  size_t GetNumDevices() const override { return 1; }

  std::vector<std::string> GetLocalDevices() const override;

  std::vector<std::string> GetAllDevices() const override;

  void SetReplicationDevices(
      std::shared_ptr<std::vector<std::string>> devices) override;

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

 private:
  std::vector<DataPtr> TransferToServerInternal(
      lazy_tensors::Span<const TensorSource> tensors);

  static lazy_tensors::client::ShapeData GetShapeData(const Shape& shape);

 private:
  Options options_;
  int compilation_cnt_{0};
  std::unordered_map<const Computation*, tvm::IRModule> lifted_computation_;

};

lazy_tensors::ComputationClient* MNMGet();

lazy_tensors::ComputationClient* MNMGetIfInitialized();

}
