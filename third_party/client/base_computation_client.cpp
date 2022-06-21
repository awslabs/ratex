/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "client/base_computation_client.h"

#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"

#include "lazy_tensors/computation_client/nnc_computation_client.h"

#include "ratex/csrc/compiler/utils.h"
#include "ratex/csrc/compiler/raf_lowering_context.h"
#include "ratex/csrc/utils/file.h"

#include "raf/serialization.h"

#include "env_vars.h"

namespace lazy_tensors {

using namespace torch_lazy_tensors::compiler;
using namespace raf;

std::once_flag g_computation_client_once;
std::atomic<lazy_tensors::ComputationClient*> g_computation_client(nullptr);

ComputationClient* ComputationClient::Get() {
  return getBackendRegistrar()->GetComputationClient();
}

std::unique_ptr<ComputationClient> ComputationClient::Create() {
  LOG(FATAL) << "NotImplemented Error";
}

}  // namespace lazy_tensors

namespace ratex {

using namespace lazy_tensors;

void PopulateLocalDevices(BaseComputationClient::Options* options) {
  auto dev_kind = sys_util::GetEnvString(ratex::env::kEnvDefaultDevice, "CPU");

  auto count = sys_util::GetEnvInt(ratex::env::kEnvDeviceCount, 0);
  CHECK_GT(count, 0) << "RATEX_DEVICE_COUNT is not set, something must be wrong!";
  if (dev_kind == "CPU" || dev_kind == "GPU") {
    for (size_t i = 0; i < count; ++i) {
      std::string ltc_device = dev_kind + ":" + std::to_string(i);
      if (i == 0) {
        options->default_device = ltc_device;
      }
      options->devices.insert(ltc_device);
      options->global_device_map[ltc_device] =
          torch_lazy_tensors::compiler::raf_backend::ToRAFDevice(ltc_device).c_str();
    }
  } else {
    LOG(FATAL) << "Unsupported device type: " << dev_kind;
  }
}

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

void BaseComputationClient::SaveArtifacts(const std::string& dir, const std::string& json) {
  std::string compute_file_path = dir + "/compute.json";
  Save(compute_file_path, json);
}

std::vector<ComputationClient::ComputationPtr> BaseComputationClient::Compile(
    std::vector<ComputationClient::CompileInstance> instances) {
  std::vector<ComputationPtr> results;
  for (const auto& ins : instances) {
    if (options_.cache_enabled) {
      static auto query = registry::GetPackedFunc("ratex.utils.cache.query");
      static auto create_entry = registry::GetPackedFunc("ratex.utils.cache.create_entry");
      static auto acquire_lock =
          registry::GetPackedFunc("ratex.utils.cache.acquire_cache_entry_lock");
      static auto release_lock =
          registry::GetPackedFunc("ratex.utils.cache.release_cache_entry_lock");

      const auto& key = CompileCacheKey(ins);
      acquire_lock(key);
      std::string dirname = query(key).operator std::string();
      if (PathExist(dirname)) {
        // Cache Hit
        release_lock(key);
        std::string compute_file = dirname + "/compute.json";
        results.push_back(CompileDeSerialize(compute_file));
      } else {
        // Cache Miss
        ComputationPtr res = Compile(ins);
        dirname = create_entry(key).operator std::string();
        std::string json = CompileSerialize(res);
        SaveArtifacts(dirname, json);
        results.push_back(res);
        release_lock(key);
      }
    } else {
      results.push_back(Compile(ins));
    }

    std::string dump_alias_path = lazy_tensors::sys_util::GetEnvString("RATEX_DUMP_ALIAS", "");
    if (!dump_alias_path.empty()) {
      DumpComputationAlias(ins, dump_alias_path);
    }
  }
  return results;
}

ObjectRef BaseComputationClient::CompileCacheKey(CompileInstance instance) {
  auto* computation =
      static_cast<torch_lazy_tensors::compiler::raf_backend::GenericComputationRAF*>(
          instance.computation.get());
  auto func = Downcast<Function>(computation->computation());
  Array<Integer> model_states;
  Map<Integer, Integer> alias;
  for (size_t i = 0; i < func->params.size(); ++i) {
    Var var = func->params[i];
    if (computation->model_states().find(var) != computation->model_states().end()) {
      model_states.push_back(i);
    }
  }
  for (const auto& kv : computation->alias()) {
    alias.Set(kv.first, kv.second);
  }

  IRModule ir_module = IRModule::FromExpr(computation->computation());
  // Canonicalize IR.
  ir_module = raf::pass::FoldConstant()(ir_module);
  ir_module = raf::pass::InferType()(ir_module);
  String json(raf::ir::serialization::SaveJSON(ir_module));

  return Array<ObjectRef>({json, model_states, alias});
}

void BaseComputationClient::DumpComputationAlias(const CompileInstance& instance,
                                                 std::string path) {
  auto* computation =
      static_cast<torch_lazy_tensors::compiler::raf_backend::GenericComputationRAF*>(
          instance.computation.get());
  std::string alias = "";
  for (const auto& kv : computation->alias()) {
    alias.append(std::to_string(kv.first) + " " + std::to_string(kv.second) + "\n");
  }
  Save(path, alias);
}

}  // namespace ratex