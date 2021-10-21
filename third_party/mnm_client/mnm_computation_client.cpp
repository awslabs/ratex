#include "mnm_client/mnm_computation_client.h"

#include <fstream>
#include <iostream>

#include "torch_mnm/csrc/compiler/utils.h"
#include "torch_mnm/csrc/compiler/mnm_lowering_context.h"
#include "torch_mnm/csrc/value_ext/value.h"
#include "torch_mnm/csrc/pass_ext/pass.h"

#include "lazy_tensors/computation_client/nnc_computation_client.h"

#include "tvm/node/serialization.h"
#include "mnm/base.h"
#include "mnm/serialization.h"
#include "mnm/vm/vm.h"
#include "mnm/vm/value.h"
#include "meta/src/common/shape_utils.h"
#include "meta/src/impl/vm/compiler.h"
#include "meta/src/op/ty/utils.h"

namespace torch_mnm {

using namespace torch_lazy_tensors::compiler::mnm_backend;

lazy_tensors::ComputationClient* CreateClient() {
  auto client = ComputationClient::Create();
  return client.release();
}

}  // namespace torch_mnm

namespace lazy_tensors {

std::once_flag g_computation_client_once;
std::atomic<lazy_tensors::ComputationClient*> g_computation_client(nullptr);

ComputationClient* ComputationClient::Get() {
  std::call_once(g_computation_client_once,
                 [&]() { g_computation_client = torch_mnm::CreateClient(); });
  return g_computation_client.load(); 
}

std::unique_ptr<ComputationClient> ComputationClient::Create() {
  torch_mnm::MNMComputationClient::Options options;
  // TODO(@hzfan): populate options like pytorch-ltc/xla/third_party/xla_client/computation_client.cc:
  // XrtComputationClient::Options options;
  // std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto;
  // if (!ParseEnvBasedTpuClusterConfig(&options) &&
  //     !ParseEnvDeviceCounts(&options) && !ParseEnvDevices(&options) &&
  //     !ParseMeshConfig(&options, &topology_proto)) {
  //   XLA_ERROR() << "Missing XLA configuration";
  // }
  // PopulateLocalDevices(&options);
  return std::unique_ptr<ComputationClient>(
      new torch_mnm::MNMComputationClient(options));
}

}  // namespace lazy_tensors


namespace torch_mnm {

void MNMComputationClient::MNMData::Assign(const Data& data) {
  const MNMData& mnm_data = dynamic_cast<const MNMData&>(data);
  if (&mnm_data != this) {
    handle = mnm_data.handle;
  }
}

MNMComputationClient::MNMComputationClient(MNMComputationClient::Options options) : options_(options) { }

template <typename NativeT>
void PopulateRn(lazy_tensors::Literal& literal, lazy_tensors::Span<const NativeT> values) {
  LTC_CHECK(literal.shape().IsArray());
  LTC_CHECK_EQ(ShapeUtil::ElementsIn(literal.shape()), values.size());
  LTC_CHECK_EQ(literal.shape().element_type(),
               primitive_util::NativeToPrimitiveType<NativeT>());
  auto data_span = literal.data<NativeT>();
  std::copy(values.begin(), values.end(), data_span.begin());
}

void PopulateRn(lazy_tensors::Literal& literal, const DLTensor* dlt) {
  DType dtype = dlt->dtype;
  switch (dtype.code) {
    case kDLInt:
      if (dtype.bits == 8) return PopulateRn(literal, Span<const lazy_tensors::int8>(
        reinterpret_cast<const lazy_tensors::int8*>(dlt->data), mnm::common::shape_utils::GetNumel(*dlt)));
      if (dtype.bits == 32) return PopulateRn(literal, Span<const lazy_tensors::int32>(
        reinterpret_cast<const lazy_tensors::int32*>(dlt->data), mnm::common::shape_utils::GetNumel(*dlt)));
      if (dtype.bits == 64) return PopulateRn(literal, Span<const lazy_tensors::int64>(
        reinterpret_cast<const lazy_tensors::int64*>(dlt->data), mnm::common::shape_utils::GetNumel(*dlt)));
      break;
    case kDLUInt:
      if (dtype.bits == 1) return PopulateRn(literal, Span<const bool>(
        reinterpret_cast<const bool*>(dlt->data), mnm::common::shape_utils::GetNumel(*dlt)));
      if (dtype.bits == 8) return PopulateRn(literal, Span<const lazy_tensors::uint8>(
        reinterpret_cast<const lazy_tensors::uint8*>(dlt->data), mnm::common::shape_utils::GetNumel(*dlt)));
      break;
    case kDLFloat:
      if (dtype.bits == 16) return PopulateRn(literal, Span<const lazy_tensors::half>(
        reinterpret_cast<const lazy_tensors::half*>(dlt->data), mnm::common::shape_utils::GetNumel(*dlt)));
      if (dtype.bits == 32) return PopulateRn(literal, Span<const float>(
        reinterpret_cast<const float*>(dlt->data), mnm::common::shape_utils::GetNumel(*dlt)));
      if (dtype.bits == 64) return PopulateRn(literal, Span<const double>(
        reinterpret_cast<const double*>(dlt->data), mnm::common::shape_utils::GetNumel(*dlt)));
      break;
  }
  LTC_LOG(FATAL) << "NotImplementedError: " << dtype.c_str();
}

ComputationClient::DataPtr MNMComputationClient::CreateDataPlaceholder(std::string device,
                                                    Shape shape) {
  return std::make_shared<MNMData>(std::move(device), shape);
}

std::vector<ComputationClient::DataPtr>
MNMComputationClient::TransferToServerInternal(
  lazy_tensors::Span<const TensorSource> tensors) {
  std::vector<mnm::value::TensorValue> tvs(tensors.size());
  std::vector<ComputationClient::DataPtr> result;
  for (const auto& ts : tensors) {
    mnm::DType dtype;
    std::vector<int64_t> shape;
    mnm::Device dev_cpu(mnm::DevType::kCPU(), 0);
    mnm::Device dev = ToMNMDevice(ts.device);
    std::tie(shape, dtype) = ToMNMShape(ts.shape);
    TensorValue tv_shape = TensorValue::Assemble(dev_cpu, dtype, shape);
    int64_t nbytes = mnm::common::shape_utils::BytesCompactTensor(*(tv_shape.operator DLTensor*()));
    auto buffer_cpu = memory_pool::Memory::Alloc(dev_cpu, nbytes);
    auto tv_cpu = TensorValue::Assemble(dev_cpu, dtype, shape, {}, buffer_cpu->data, buffer_cpu);
    ts.populate_fn(ts, buffer_cpu->data, nbytes);
    auto tv = TensorValue::make(mnm::tensor::Tensor(tv_cpu->tensor.CopyTo(dev)));  // memory of tv is allocated by tvm
    result.push_back(std::make_shared<MNMComputationClient::MNMData>(
      ts.device,
      Shape(ts.shape),
      tv
    ));
  }
  return result;
}

std::vector<ComputationClient::DataPtr> MNMComputationClient::TransferToServer(
    lazy_tensors::Span<const TensorSource> tensors) {
  // TODO(@hzfan): parallel transfer
  return TransferToServerInternal(tensors);
}

std::vector<Literal> MNMComputationClient::TransferFromServer(
    lazy_tensors::Span<const DataPtr> handles) {
  std::vector<Literal> results;
  for (const auto& handle : handles) {
    auto* ptr = static_cast<MNMData*>(handle.get());
    DLTensor* val = ptr->handle;
    Literal res(ToLTCShape(
      std::vector<int64_t>(val->shape, val->shape + val->ndim),
      val->dtype
    ));
    PopulateRn(res, val);
    results.push_back(res);
  }
  return results;
}

bool IsIdentityFunction(Function func) {
  if (func->params.size() != 1U)  return false;
  if (func->body != func->params[0])  return false;
  return true;
}

std::vector<ComputationClient::ComputationPtr> MNMComputationClient::Compile(
      std::vector<ComputationClient::CompileInstance> instances) {
  std::vector<ComputationPtr> results;
  for (const auto& ins : instances) {
    mnm::executor::vm::VMCompiler compiler;
    auto* computation = static_cast<GenericComputationMNM*>(ins.computation.get());
    Function func = Downcast<Function>(computation->computation());
    IRModule ir_module = IRModule::FromExpr(computation->computation());
    // std::cout << "Compile: " << std::endl;
    // std::cout << ::mnm::ir::AsText(ir_module) << std::endl;
    // std::cout << "Alias: " << std::endl;
    // for (const auto& kv : computation->alias()) {
    //   std::cout << "(" << kv.first << ", " << kv.second << "), ";
    // }
    // std::cout << std::endl;
    ir_module = mnm::pass::InferType()(ir_module);
    ir_module = mnm::pass::LambdaLift()(ir_module);
    ir_module = mnm::pass::InferType()(ir_module);
    tvm::runtime::Module exe;
    if (!IsIdentityFunction(func)) {
      // TODO(@hzfan): difference between compilation_device and devices
      // TODO(@hzfan): device convert string->tvm TargetsMap
      // TODO(@hzfan): calculate target and target_host
      // ToMNMDevice(computation->devices[0]);
      // std::cout << "Compile: " << std::endl;
      // std::cout << ::mnm::ir::AsText(ir_module) << std::endl;
      tvm::Target target_host("llvm");
      mnm::executor::vm::TargetsMap target{
        {Integer((int)(DLDeviceType::kDLCPU)), tvm::Target("llvm")}
      };
      compiler.Lower(ir_module, target, target_host);
      exe = compiler.GetFunction("get_executable", nullptr)();
    }
    results.emplace_back(std::make_shared<MNMComputation>(
      ins.computation,
      ConsumeValue(ins.computation->GetProgramShape()),
      ins.devices,
      exe
    ));
    lifted_computation_[results.back().get()] = ir_module;
  }
  return results;
}

std::string slurp(std::ifstream& in) {
    std::ostringstream sstr;
    sstr << in.rdbuf();
    return sstr.str();
}

void ExecuteCMD(std::string cmd) {
  std::cout << "+ " << cmd << std::endl;
  LTC_CHECK(0 == system(cmd.c_str()));
}


std::vector<ComputationClient::DataPtr>
MNMComputationClient::ExecuteComputation(
    const Computation& computation, lazy_tensors::Span<const DataPtr> arguments,
    const std::string& device, const ExecuteComputationOptions& options) {
  std::function<std::vector<ComputationClient::DataPtr>(Value)> explode_tuple =
    [&](Value val)->std::vector<ComputationClient::DataPtr> {
    if (const auto* tup = val.as<TupleValueObj>()) {
      std::vector<ComputationClient::DataPtr> ret;
      for (const auto& field : tup->fields) {
        std::vector<ComputationClient::DataPtr> tup_ret = explode_tuple(field);
        LTC_CHECK_EQ(tup_ret.size(), 1U);
        ret.push_back(tup_ret[0]);
      }
      return ret;
    } else if (const auto* tensor = val.as<TensorValueObj>()) {
      Shape shape = ToLTCShape(mnm::op::GetType(val));
      return {std::make_shared<MNMData>(device, shape, val)};
    }
    else if (const auto* closure_val_ext = val.as<ClosureValueExtObj>()) {
      auto func = Downcast<Function>(closure_val_ext->mod->Lookup(closure_val_ext->gvar));
      Shape shape = ToLTCShape(func->checked_type());
      return {std::make_shared<MNMData>(device, shape, val)};
    } else if (val.as<mnm::executor::vm::VMClosureValueObj>()) {
      // the return type of VMClosureValue cannot be inferred from its value solely
      return {std::make_shared<MNMData>(device, Shape(), val)};
    }
    LTC_LOG(FATAL) << "NotImplementedError: " << val->GetTypeKey();
  };
  std::function<Value(Value)> normalize_value =
    [&](Value val)->Value {
    if (const auto* vm_closure_val = val.as<mnm::executor::vm::VMClosureValueObj>()) {
      IRModule mod = lifted_computation_.at(&computation);
      const auto& mnm_computation = static_cast<const MNMComputation&>(computation);
      std::string gvar_name;
      bool found = false;
      const auto* executable = mnm_computation.executable.as<mnm::executor::vm::Executable>();
      LTC_CHECK(executable);
      for (const auto& kv : executable->global_map) {
        if (kv.second == vm_closure_val->func_index) {
          gvar_name = kv.first;
          found = true;
          break;
        }
      }
      CHECK(found);
      GlobalVar gvar = mod->GetGlobalVar(gvar_name);
      ir::Map<ir::Var, Value> env;
      auto func = Downcast<Function>(mod->Lookup(gvar_name));
      size_t num_free_vars = func->params.size();
      CHECK_EQ(func->params.size(), vm_closure_val->free_vars.size());
      for (size_t i = 0; i < num_free_vars; ++i) {
        env.Set(func->params[i], vm_closure_val->free_vars[i]);
      }
      return ClosureValueExt::make(env, mod, gvar);
    }
    return val;
  };
  static auto vm_constructor = registry::GetPackedFunc("mnm.vm.VirtualMachine");
  const auto& mnm_computation = static_cast<const MNMComputation&>(computation);
  bool is_identity_function = !mnm_computation.executable.defined();
  std::vector<Value> values;
  Value ret;
  for (const auto& argument : arguments) {
    values.push_back(static_cast<MNMData*>(argument.get())->handle);
  }
  if (!is_identity_function) {
    // TODO(@hzfan): cache the VM
    tvm::runtime::Module vm_module = mnm_computation.executable.defined() ? vm_constructor(mnm_computation.executable, false) : tvm::runtime::Module();
    auto* vm = dynamic_cast<mnm::executor::vm::VirtualMachine*>(vm_module.operator->());
    // TODO(@hzfan): feed set_devices with user-configured LTC device
    vm_module->GetFunction("set_devices")(ToMNMDevice(""));
    mnm::executor::vm::VMContext vm_ctx = vm->PrepareVMContext("main", values);
    // TODO(@hzfan): sync the execution
    ret = vm->Run(vm_ctx);
  } else {
    LTC_CHECK_EQ(values.size(), 1U);
    ret = values[0];
  }
  ret = normalize_value(ret);
  return explode_tuple(ret);
}

std::string MNMComputationClient::GetResourceDomain(
    const std::string& device) const {
  return "";
}

std::string MNMComputationClient::GetDefaultDevice() const {
  switch (lazy_tensors::NNCComputationClient::HardwareDeviceType()) {
    case at::kCPU: {
      return "CPU:0";
    }
    case at::kCUDA: {
      return "GPU:0";
    }
    default: { LTC_LOG(FATAL) << "Invalid device type"; }
  }
}

std::vector<std::string> MNMComputationClient::GetLocalDevices() const {
  return {GetDefaultDevice()};
}

std::vector<std::string> MNMComputationClient::GetAllDevices() const {
  return GetLocalDevices();
}

void MNMComputationClient::SetReplicationDevices(
    std::shared_ptr<std::vector<std::string>> devices) {
  LTC_CHECK_EQ(devices->size(), size_t(1)) << "Replication not supported yet";
}

std::shared_ptr<std::vector<std::string>>
MNMComputationClient::GetReplicationDevices() {
  return nullptr;
}

void MNMComputationClient::PrepareToExit() {}

lazy_tensors::client::ShapeData MNMComputationClient::GetShapeData(
    const Shape& shape) {
  std::vector<int64_t> dimensions(shape.dimensions().begin(),
                                  shape.dimensions().end());
  lazy_tensors::PrimitiveType element_type = shape.element_type();
  std::vector<lazy_tensors::client::ShapeData> element_shapes;
  for (const Shape& element_shape : shape.tuple_shapes()) {
    element_shapes.push_back(GetShapeData(element_shape));
  }
  auto minor_to_major = shape.layout().minor_to_major();
  return lazy_tensors::client::ShapeData(
      element_type, dimensions, element_shapes,
      std::vector<int64_t>(minor_to_major.begin(), minor_to_major.end()));
}

lazy_tensors::ComputationClient* MNMGet() {
  using namespace lazy_tensors;
  std::call_once(g_computation_client_once,
                 [&]() { g_computation_client = CreateClient(); });
  return g_computation_client.load();
}

lazy_tensors::ComputationClient* MNMGetIfInitialized() {
  using namespace lazy_tensors;
  return g_computation_client.load();
}

}  // namespace torch_mnm
