/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "client/raf_computation_client.h"

#include <fstream>
#include <iostream>

#include "ratex/csrc/compiler/utils.h"
#include "ratex/csrc/compiler/raf_lowering_context.h"
#include "ratex/csrc/raf_model_state.h"
#include "ratex/csrc/value_ext/value.h"
#include "ratex/csrc/pass_ext/pass.h"
#include "ratex/csrc/utils/file.h"
#include "env_vars.h"

#include "lazy_tensors/computation_client/nnc_computation_client.h"
#include "lazy_tensor_core/csrc/device.h"

#include "tvm/node/serialization.h"
#include "raf/device.h"
#include "raf/pass_manager.h"
#include "raf/serialization.h"
#include "raf/vm/vm.h"
#include "raf/vm/value.h"
#include "raf/src/common/shape_utils.h"
#include "raf/src/impl/vm/compiler.h"
#include "raf/src/op/ty/utils.h"

namespace ratex {

using namespace torch_lazy_tensors::compiler;
using namespace torch_lazy_tensors::compiler::raf_backend;
using namespace raf::value;

void RAFComputationClient::RAFData::Assign(const Data& data) {
  const RAFData& raf_data = dynamic_cast<const RAFData&>(data);
  if (&raf_data != this) {
    handle = raf_data.handle;
  }
}

RAFComputationClient::RAFComputationClient(Options options) : BaseComputationClient(options) {
}

std::unique_ptr<ComputationClient> RAFComputationClient::Create() {
  Options options;
  PopulateLocalDevices(&options);
  return std::make_unique<RAFComputationClient>(options);
}

template <typename NativeT>
void PopulateRn(lazy_tensors::Literal& literal, lazy_tensors::Span<const NativeT> values) {
  LTC_CHECK(literal.shape().IsArray());
  LTC_CHECK_EQ(ShapeUtil::ElementsIn(literal.shape()), values.size());
  LTC_CHECK_EQ(literal.shape().element_type(), primitive_util::NativeToPrimitiveType<NativeT>());
  auto data_span = literal.data<NativeT>();
  std::copy(values.begin(), values.end(), data_span.begin());
}

void PopulateRn(lazy_tensors::Literal& literal, void* buf) {
  using namespace lazy_tensors;
  switch (literal.shape().element_type()) {
    case PrimitiveType::S8:
      return PopulateRn(literal, Span<const int8_t>(reinterpret_cast<const int8_t*>(buf),
                                                    literal.value().numel()));
    case PrimitiveType::S32:
      return PopulateRn(literal, Span<const int32_t>(reinterpret_cast<const int32_t*>(buf),
                                                     literal.value().numel()));
    case PrimitiveType::S64:
      return PopulateRn(literal, Span<const int64_t>(reinterpret_cast<const int64_t*>(buf),
                                                     literal.value().numel()));
    case PrimitiveType::PRED:
      return PopulateRn(
          literal, Span<const bool>(reinterpret_cast<const bool*>(buf), literal.value().numel()));
    case PrimitiveType::U8:
      return PopulateRn(literal, Span<const uint8_t>(reinterpret_cast<const uint8_t*>(buf),
                                                     literal.value().numel()));
    case PrimitiveType::U32:
      return PopulateRn(literal, Span<const uint32_t>(reinterpret_cast<const uint32_t*>(buf),
                                                      literal.value().numel()));
    case PrimitiveType::U64:
      return PopulateRn(literal, Span<const uint64_t>(reinterpret_cast<const uint64_t*>(buf),
                                                      literal.value().numel()));
    case PrimitiveType::F16:
      return PopulateRn(literal, Span<const at::Half>(reinterpret_cast<const at::Half*>(buf),
                                                      literal.value().numel()));
    case PrimitiveType::F32:
      return PopulateRn(
          literal, Span<const float>(reinterpret_cast<const float*>(buf), literal.value().numel()));
    case PrimitiveType::F64:
      return PopulateRn(literal, Span<const double>(reinterpret_cast<const double*>(buf),
                                                    literal.value().numel()));
    case PrimitiveType::BF16:
      return PopulateRn(literal,
                        Span<const at::BFloat16>(reinterpret_cast<const at::BFloat16*>(buf),
                                                 literal.value().numel()));
    default:
      LTC_LOG(FATAL) << "NotImplementedError: " << literal.shape().element_type();
  }
}

ComputationClient::DataPtr RAFComputationClient::CreateDataPlaceholder(std::string device,
                                                                       Shape shape) {
  return std::make_shared<RAFData>(std::move(device), shape);
}

std::vector<ComputationClient::DataPtr> RAFComputationClient::TransferToServerInternal(
    lazy_tensors::Span<const TensorSource> tensors) {
  std::vector<raf::value::TensorValue> tvs(tensors.size());
  std::vector<ComputationClient::DataPtr> result;
  for (const auto& ts : tensors) {
    raf::DType dtype;
    std::vector<int64_t> shape;
    raf::Device dev_cpu(raf::DevType::kCPU(), 0);
    raf::Device dev = ToRAFDevice(ts.device);
    std::tie(shape, dtype) = ToRAFShape(ts.shape);
    TensorValue tv_shape = raf::value::TensorValue::Assemble(dev_cpu, dtype, shape);
    int64_t nbytes = raf::common::shape_utils::BytesCompactTensor(*(tv_shape.operator DLTensor*()));
    auto buffer_cpu = raf::memory_pool::Memory::Alloc(dev_cpu, nbytes);
    auto tv_cpu = TensorValue::Assemble(dev_cpu, dtype, shape, {}, buffer_cpu->data, buffer_cpu);
    ts.populate_fn(ts, buffer_cpu->data, nbytes);
    auto tv = TensorValue::make(
        raf::tensor::Tensor(tv_cpu->tensor.CopyTo(dev)));  // memory of tv is allocated by tvm
    result.push_back(
        std::make_shared<RAFComputationClient::RAFData>(ts.device, Shape(ts.shape), tv));
  }
  return result;
}

std::vector<ComputationClient::DataPtr> RAFComputationClient::TransferToServer(
    lazy_tensors::Span<const TensorSource> tensors) {
  // TODO(@hzfan): parallel transfer
  return TransferToServerInternal(tensors);
}

std::vector<Literal> RAFComputationClient::TransferFromServer(
    lazy_tensors::Span<const DataPtr> handles) {
  std::vector<Literal> results;
  for (const auto& handle : handles) {
    auto* ptr = static_cast<RAFData*>(handle.get());
    DLTensor* val = ptr->handle;
    auto shape = std::vector<int64_t>(val->shape, val->shape + val->ndim);
    Literal res(ToLTCShape(shape, val->dtype));

    // Transfer to CPU if it is on the other device.
    if (val->device.device_type != DevType::kCPU()) {
      raf::Device dev_cpu(raf::DevType::kCPU(), 0);
      TensorValue tv_shape = TensorValue::Assemble(dev_cpu, val->dtype, shape);
      int64_t nbytes =
          raf::common::shape_utils::BytesCompactTensor(*(tv_shape.operator DLTensor*()));
      auto buffer_cpu = memory_pool::Memory::Alloc(dev_cpu, nbytes);
      auto tv_cpu =
          TensorValue::Assemble(dev_cpu, val->dtype, shape, {}, buffer_cpu->data, buffer_cpu);
      tv_cpu->tensor.CopyFrom(val);
      PopulateRn(res, (tv_cpu.operator DLTensor*())->data);
    } else {
      PopulateRn(res, val->data);
    }
    results.push_back(res);
  }
  return results;
}

bool IsIdentityFunction(Function func) {
  if (func->params.size() != 1U) return false;
  if (func->body != func->params[0]) return false;
  return true;
}

ComputationClient::ComputationPtr RAFComputationClient::Compile(
    ComputationClient::CompileInstance instance) {
  LTC_TIMED("RAFCompile");
  bool is_amp_enabled = torch_lazy_tensors::GetRAFModelState()->IsAMPEnabled();
  auto* computation = static_cast<GenericComputationRAF*>(instance.computation.get());
  Function func = Downcast<Function>(computation->computation());
  IRModule ir_module = IRModule::FromExpr(computation->computation());

  tvm::runtime::Module exe, vm_module;
  if (!IsIdentityFunction(func)) {
    // For uncached function, we perform the VM compilation and cache the VM.
    // Note that ops in the VM are not JITed until the first execution, but
    // we still need to cache the VM to reuse the JITed ops.
    raf::executor::vm::VMCompiler compiler;

    // Build the alias map.
    ir::Map<tvm::Integer, tvm::Integer> alias_map;
    for (const auto& kv : computation->alias()) {
      alias_map.Set(kv.first, kv.second);
    }

    auto raf_device = ToRAFDevice(instance.compilation_device);

    raf::pass::RAFSequential seq(
        {
            raf::pass::InferType(),
            raf::pass::FoldConstant(),
            raf::pass::DeadCodeElimination(),
            raf::pass::InferType(),
            raf::pass::SimplifyExpr(),
            raf::pass::InferType(),
            raf::pass::DeadCodeElimination(),
            raf::pass::InferType(),
            raf::pass::LambdaLift(),
            raf::pass::InferType(),
            raf::pass::InlineClosure(),
            raf::pass::InferType(),
            raf::pass::DeadCodeElimination(),
            raf::pass::InferType(),
            raf::pass::EliminateClosure(),
            raf::pass::InferType(),
            raf::pass::InlineLet(),
            raf::pass::InferType(),
            raf::pass::DeadCodeElimination(),
            raf::pass::InferType(),
            raf::pass::CanonicalizeOps(),
            raf::pass::InferType(),
            raf::pass::AssignDevice(raf_device.c_str()),
        },
        "ratex_raf_compile");

    raf::executor::vm::DeviceMap device_map{{Integer((int)(raf_device.device_type())), raf_device}};

    // Rematerialization will be enabled if memory budget > 0.
    auto memory_budget = lazy_tensors::sys_util::GetEnvInt("RATEX_MEMORY_BUDGET", 0);

    auto pass_ctx = pass::PassContext::Create();
    pass_ctx->opt_level = 3;
    pass_ctx->config.Set("raf.amp.out_dtype", String("float32"));
    pass_ctx->config.Set("raf.memory_schedule", Bool(true));
    pass_ctx->config.Set("raf.memory_budget", Integer(IntImm(DataType::Int(64), memory_budget)));
    pass_ctx->config.Set("raf.remat.use_gflops_cost", Bool(false));
    {
      tvm::With<pass::PassContext> ctx_scope(pass_ctx);
      tvm::With<raf::Device> dev_ctx(raf_device);
      if (!alias_map.empty()) {
        ir_module = raf::pass::InplaceUpdateByAlias(alias_map)(ir_module);
      }
      ir_module = seq(ir_module);
      ir_module = IRModule::FromExpr(ir_module->Lookup("main"));
      ir_module = raf::pass::InferType()(ir_module);
      if (is_amp_enabled) {
        ir_module = raf::pass::AutoCast()(ir_module);
      }
      compiler.Lower(ir_module, device_map);
    }
    exe = compiler.GetFunction("get_executable", nullptr)();

    static auto vm_constructor = registry::GetPackedFunc("raf.vm.VirtualMachine");
    vm_module = vm_constructor(exe, false, false);
    vm_module->GetFunction("set_devices")(raf_device);
  }
  auto ret = std::make_shared<RAFComputation>(instance.computation,
                                              ConsumeValue(instance.computation->GetProgramShape()),
                                              instance.devices, exe, vm_module);
  lifted_computation_[ret.get()] = ir_module;

  std::string file_path = lazy_tensors::sys_util::GetEnvString("RATEX_SAVE_IR_FILE", "");
  if (file_path != "") {
    Save(file_path, raf::serialization::SaveJSON(ir_module));
  }

  return ret;
}

std::vector<ComputationClient::DataPtr> RAFComputationClient::ExecuteComputation(
    const Computation& computation, lazy_tensors::Span<const DataPtr> arguments,
    const std::string& device, const ExecuteComputationOptions& options) {
  LTC_TIMED("RAFExecute");

  bool is_dryrun = lazy_tensors::sys_util::GetEnvBool("RATEX_DRY_RUN", false);
  if (is_dryrun) {
    return DryrunComputation(computation, arguments, device, options);
  }

  std::function<std::vector<ComputationClient::DataPtr>(Value)> explode_tuple =
      [&](Value val) -> std::vector<ComputationClient::DataPtr> {
    if (const auto* tup = val.as<TupleValueObj>()) {
      std::vector<ComputationClient::DataPtr> ret;
      for (const auto& field : tup->fields) {
        std::vector<ComputationClient::DataPtr> tup_ret = explode_tuple(field);
        LTC_CHECK_EQ(tup_ret.size(), 1U);
        ret.push_back(tup_ret[0]);
      }
      return ret;
    } else if (const auto* tensor = val.as<TensorValueObj>()) {
      Shape shape = ToLTCShape(raf::op::GetType(val));
      return {std::make_shared<RAFData>(device, shape, val)};
    } else if (const auto* closure_val_ext = val.as<ClosureValueExtObj>()) {
      auto func = Downcast<Function>(closure_val_ext->mod->Lookup(closure_val_ext->gvar));
      Shape shape = ToLTCShape(func->checked_type());
      return {std::make_shared<RAFData>(device, shape, val)};
    } else if (val.as<raf::executor::vm::VMClosureValueObj>()) {
      // the return type of VMClosureValue cannot be inferred from its value solely
      return {std::make_shared<RAFData>(device, Shape(), val)};
    }
    LTC_LOG(FATAL) << "NotImplementedError: " << val->GetTypeKey();
  };
  std::function<Value(Value)> normalize_value = [&](Value val) -> Value {
    if (const auto* vm_closure_val = val.as<raf::executor::vm::VMClosureValueObj>()) {
      IRModule mod = lifted_computation_.at(&computation);
      const auto& raf_computation = static_cast<const RAFComputation&>(computation);
      std::string gvar_name;
      bool found = false;
      const auto* executable = raf_computation.executable.as<raf::executor::vm::Executable>();
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
  const auto& raf_computation = static_cast<const RAFComputation&>(computation);
  bool is_identity_function = !raf_computation.executable.defined();
  std::vector<Value> values;
  Value ret;
  for (const auto& argument : arguments) {
    values.push_back(static_cast<RAFData*>(argument.get())->handle);
  }
  if (!is_identity_function) {
    auto vm_module = raf_computation.vm_module;
    auto* vm = dynamic_cast<raf::executor::vm::VirtualMachine*>(vm_module.operator->());

    // Enable auto scheduler when JITing kernels at the first run.
    static auto pass_ctx = pass::PassContext::Create();
    pass_ctx->config.Set("relay.backend.use_auto_scheduler", Bool(true));
    {
      tvm::With<pass::PassContext> ctx_scope(pass_ctx);
      raf::executor::vm::VMContext vm_ctx = vm->PrepareVMContext("main", values);
      ret = vm->Run(vm_ctx);  // TODO(@hzfan): sync the execution
    }
  } else {
    LTC_CHECK_EQ(values.size(), 1U);
    ret = values[0];
  }
  ret = normalize_value(ret);
  return explode_tuple(ret);
}

TensorValue MakeZeros(Type ty, std::string device) {
  auto tty = Downcast<TensorType>(ty);
  raf::Device dev_cpu(raf::DevType::kCPU(), 0);
  raf::Device dev = ToRAFDevice(device);
  raf::DType dtype(tty->dtype.operator DLDataType());
  std::vector<int64_t> shape(raf::op::ArrayToInt(tty->shape));
  TensorValue tv_shape = raf::value::TensorValue::Assemble(dev_cpu, dtype, shape);
  int64_t nbytes = raf::common::shape_utils::BytesCompactTensor(*(tv_shape.operator DLTensor*()));
  auto buffer_cpu = raf::memory_pool::Memory::Alloc(dev_cpu, nbytes);
  std::memset(buffer_cpu->data, 0, nbytes);
  auto tv_cpu = TensorValue::Assemble(dev_cpu, dtype, shape, {}, buffer_cpu->data, buffer_cpu);
  auto tv = TensorValue::make(
      raf::tensor::Tensor(tv_cpu->tensor.CopyTo(dev)));  // memory of tv is allocated by tvm
  return tv;
}

std::vector<ComputationClient::DataPtr> RAFComputationClient::DryrunComputation(
    const Computation& computation, lazy_tensors::Span<const DataPtr> arguments,
    const std::string& device, const ExecuteComputationOptions& options) {
  const auto& raf_computation = static_cast<const RAFComputation&>(computation);
  bool is_identity_function = !raf_computation.executable.defined();

  if (!is_identity_function) {
    IRModule mod = lifted_computation_.at(&computation);
    auto func = Downcast<Function>(mod->Lookup("main"));

    const auto& type = Downcast<FuncType>(func->checked_type())->ret_type;
    if (const auto* tty = type.as<TupleTypeNode>()) {
      std::vector<ComputationClient::DataPtr> ret;
      for (const auto& ty : tty->fields) {
        ret.push_back(std::make_shared<RAFData>(device, ToLTCShape(ty), MakeZeros(ty, device)));
      }
      return ret;
    } else if (type.as<TensorTypeNode>()) {
      return {std::make_shared<RAFData>(device, ToLTCShape(type), MakeZeros(type, device))};
    }
    LTC_LOG(FATAL) << "NotImplementedError: " << type;
  } else {
    LTC_CHECK_EQ(arguments.size(), 1U);
    return {arguments[0]};
  }
  return {};
}

lazy_tensors::ComputationClient* RAFGet() {
  using namespace lazy_tensors;
  static auto raf_computation_client = RAFComputationClient::Create();
  return raf_computation_client.get();
}

lazy_tensors::ComputationClient* RAFGetIfInitialized() {
  using namespace lazy_tensors;
  return RAFGet();
}

}  // namespace ratex
