#include "client/base_computation_client.h"

#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"

#include "lazy_tensors/computation_client/nnc_computation_client.h"

namespace lazy_tensors {

using namespace torch_lazy_tensors::compiler;

std::once_flag g_computation_client_once;
std::atomic<lazy_tensors::ComputationClient*> g_computation_client(nullptr);

ComputationClient* ComputationClient::Get() {
  return getBackendRegistrar()->GetComputationClient();
}

std::unique_ptr<ComputationClient> ComputationClient::Create() {
  // TODO(@hzfan): populate options like
  // pytorch-ltc/xla/third_party/xla_client/computation_client.cc: XrtComputationClient::Options
  // options; std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto; if
  // (!ParseEnvBasedTpuClusterConfig(&options) &&
  //     !ParseEnvDeviceCounts(&options) && !ParseEnvDevices(&options) &&
  //     !ParseMeshConfig(&options, &topology_proto)) {
  //   XLA_ERROR() << "Missing XLA configuration";
  // }
  // PopulateLocalDevices(&options);
  // Method 1: get the computation client without creating it.
  // Problem: it has already been owned, so we cannot own it with another unique_ptr
  // return getBackendRegistrar()->GetComputationClient();
  LOG(FATAL) << "NotImplemented Error";
}

}  // namespace lazy_tensors

namespace torch_mnm {

using namespace lazy_tensors;

client::ShapeData BaseComputationClient::GetShapeData(const Shape& shape) {
  std::vector<int64_t> dimensions(shape.dimensions().begin(), shape.dimensions().end());
  PrimitiveType element_type = shape.element_type();
  std::vector<client::ShapeData> element_shapes;
  for (const Shape& element_shape : shape.tuple_shapes()) {
    element_shapes.push_back(GetShapeData(element_shape));
  }
  auto minor_to_major = shape.layout().minor_to_major();
  return client::ShapeData(element_type, dimensions, element_shapes,
                           std::vector<int64_t>(minor_to_major.begin(), minor_to_major.end()));
}

std::string BaseComputationClient::GetResourceDomain(const std::string& device) const {
  return "";
}

std::string BaseComputationClient::GetDefaultDevice() const {
  // TODO(@hzfan): Investigate whether we should use the LTC API to get the default device.
  // i.e., lazy_tensors::NNCComputationClient::HardwareDeviceType()
  return options_.default_device;
}

std::vector<std::string> BaseComputationClient::GetLocalDevices() const {
  return std::vector<std::string>(options_.devices.begin(), options_.devices.end());
}

std::vector<std::string> BaseComputationClient::GetAllDevices() const {
  std::vector<std::string> devices;
  for (const auto& dev_target : options_.global_device_map) {
    devices.push_back(dev_target.first);
  }
  return devices;
}

void BaseComputationClient::SetReplicationDevices(
    std::shared_ptr<std::vector<std::string>> devices) {
  LTC_CHECK_EQ(devices->size(), size_t(1)) << "Replication not supported yet";
}

std::shared_ptr<std::vector<std::string>> BaseComputationClient::GetReplicationDevices() {
  return nullptr;
}

void BaseComputationClient::PrepareToExit() {
}

}  // namespace torch_mnm