/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "torch/csrc/jit/python/pybind.h"
#include "torch/csrc/utils/cuda_lazy_init.h"
#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/shape.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "razor/csrc/aten_mnm_bridge.h"
#include "razor/csrc/ops/all_gather.h"
#include "razor/csrc/ops/relay_expr.h"
#include "razor/csrc/ops/relay_function.h"
#include "razor/csrc/ops/mnm_ops.h"
#include "razor/csrc/compiler/utils.h"
#include "razor/csrc/mnm_model_state.h"
#include "razor/csrc/aten_mnm_bridge.h"
#include "client/mnm_computation_client.h"
#include "mnm/registry.h"
#include "meta/src/op/ty/utils.h"

namespace torch_lazy_tensors {

extern void InitLtcBindings(py::module m);

namespace {

using namespace ir;

std::vector<std::vector<lazy_tensors::int64>> CreateReduceGroups(const py::list& groups) {
  std::vector<std::vector<lazy_tensors::int64>> replica_groups;
  for (auto& group : groups) {
    replica_groups.emplace_back();
    for (auto& replica_id : group.cast<py::list>()) {
      replica_groups.back().push_back(replica_id.cast<lazy_tensors::int64>());
    }
  }
  return replica_groups;
}

Device GetDeviceOrCurrent(const std::string& device_str) {
  if (device_str.empty()) {
    return GetCurrentDevice();
  }
  return bridge::AtenDeviceToLtcDevice(c10::Device(device_str));
}

std::shared_ptr<ir::Value> CreateToken(const std::string& device_str) {
  // This should be using lazy_tensors::CreateToken() once we have added Token
  // support to the backend AllReduce(). Since mnm AllReduce doesn't need token
  // so we only return a dummy IR value.
  Device device = GetDeviceOrCurrent(device_str);
  ir::Value ir_value =
      LazyTensor::GetIrValueForScalar(0.0, lazy_tensors::PrimitiveType::F32, device);
  return std::make_shared<ir::Value>(std::move(ir_value));
}

void InitMNMModuleBindings(py::module m) {
  m.def("_mnm_invoke_relay",
        [](at::Tensor func, const std::vector<at::Tensor>& tensors,
           const std::unordered_map<int, int>& inplace_update_out_2_arg_idxs)
            -> std::vector<at::Tensor> {
          LTC_COUNTER("_mnm_invoke_relay", 1);
          LTC_CHECK_GT(tensors.size(), 0U);
          LazyTensor lazy_tensor_func = bridge::GetLtcTensor(func);
          ir::Value func_value = lazy_tensor_func.GetIrValue();
          std::vector<LazyTensor> lazy_tensors{lazy_tensor_func};
          std::vector<ir::Value> input_values{func_value};
          for (const auto& tensor : tensors) {
            LazyTensor lt = bridge::GetLtcTensor(tensor);
            lazy_tensors.push_back(lt);
            input_values.push_back(lt.GetIrValue());
          }

          // func_value should be DeviceData
          // TODO(@hzfan): use LazyTensor::Create and fix device and dtype
          // TODO(@hzfan): handle the case where fwd result is a tuple
          // bwd is closure, whose type cannot be expressed as at::ScalarType. Byte is used as dummy
          // data type for it.
          Device dev(lazy_tensors::ComputationClient::Get()->GetDefaultDevice());
          // RelayExpr returns multiple nodes
          ir::NodePtr ret = ir::MakeNode<ir::ops::RelayExpr>(input_values);
          if (ret->shape().IsTuple()) {
            std::vector<at::Tensor> unpacked_ret;
            for (int i = 0; i < ret->shape().tuple_shapes_size(); ++i) {
              const at::ScalarType tuple_type = at::ScalarType::Byte;
              at::ScalarType scalar_type =
                  ret->shape().tuple_shapes(i).IsTuple()
                      ? tuple_type
                      : TensorTypeFromLtcType(ret->shape().tuple_shapes(i).element_type());

              if (inplace_update_out_2_arg_idxs.count(i) > 0) {
                // The output inplace updates an input, so we mark it accordingly without returning.
                // FIXME(@comaniac): This inplace update is incorrect and will result in crashing
                // for ResNet-50 at the 3rd epoch.
                // LazyTensor::Create(ir::Value(ret, i), dev, scalar_type)
                //     .ShallowCopyTo(&lazy_tensors[inplace_update_out_2_arg_idxs.at(i)]);
              } else {
                unpacked_ret.emplace_back(bridge::AtenFromLtcTensor(
                    LazyTensor::Create(ir::Value(ret, i), dev, scalar_type)));
              }
            }
            return unpacked_ret;
          } else {
            return {bridge::AtenFromLtcTensor(LazyTensor::Create(
                ir::Value(ret, 0), dev, TensorTypeFromLtcType(ret->shape().element_type())))};
          }
        });

  m.def("_mnm_to_tensor", [](int64_t handle) -> at::Tensor {
    static auto handle_to_value = mnm::registry::GetPackedFunc("mnm.value.HandleToValue");
    // TODO(@hzfan): assign real data type when handle is TensorValue
    mnm::value::Value val = handle_to_value(handle);
    lazy_tensors::Shape shape =
        compiler::mnm_backend::ToLTCShape(tvm::relay::TupleType({mnm::op::GetType(val)}));
    LazyTensor ret;
    if (const auto* cvo = val.as<mnm::value::ClosureValueObj>()) {
      // ret is closure, whose type cannot be expressed as at::ScalarType. Byte is used as dummy
      // data type for it.
      LTC_CHECK_EQ(cvo->env.size(), 0U);
      Device dev(lazy_tensors::ComputationClient::Get()->GetDefaultDevice());
      ir::Value relay_function = ir::MakeNode<ir::ops::RelayFunction>(cvo->func);
      ret = LazyTensor::Create(relay_function, dev, at::ScalarType::Byte);
    } else {
      LTC_LOG(FATAL) << "Unsupported type " << val->GetTypeKey();
    }
    return bridge::AtenFromLtcTensor(ret);
  });

  // TODO: Remove this function and use LTC binding after we have control over LTC
  m.def("_mnm_all_gather",
        [](const at::Tensor& input, lazy_tensors::int64 dim, const py::list& groups) {
          std::vector<std::vector<lazy_tensors::int64>> replica_groups = CreateReduceGroups(groups);
          const LazyTensor& input_ltc = bridge::GetLtcTensor(input);
          std::vector<ir::Value> input_values({input_ltc.GetIrValue()});
          ir::NodePtr node =
              ir::MakeNode<ir::ops::MNMAllGather>(input_values, dim, std::move(replica_groups));
          LazyTensor ret = bridge::mnm_backend::CreateFrom(input_ltc, ir::Value(node, 0));
          return bridge::AtenFromLtcTensor(std::move(ret));
        });

  m.def("_mnm_create_token", [](const std::string& device) { return CreateToken(device); });

  m.def("_mnm_mark_parameter", [](at::Tensor tensor) -> at::Tensor {
    LazyTensor lazy_tensor = bridge::GetLtcTensor(tensor);
    ir::Value ir_value = lazy_tensor.GetIrValue();
    GetMNMModelState()->AddModelState(lazy_tensor);

    Value value = lazy_tensor.GetIrValue();
    lazy_tensors::ComputationClient::DataPtr data = bridge::mnm_backend::GetData(value);
    static_cast<razor::BaseComputationClient::BaseData*>(data.get())->is_param = true;

    return tensor;
  });

  m.def("_mnm_set_amp_enabled", [](bool enable) { GetMNMModelState()->SetAMPEnabled(enable); });

  m.def("_mnm_is_amp_enabled", []() { return GetMNMModelState()->IsAMPEnabled(); });

  m.def("_mnm_ltc_timed_metric", [](const std::string& name, float value) {
    lazy_tensors::metrics::Metric(name, lazy_tensors::metrics::MetricFnTime).AddSample(value);
  });

  m.def("_mnm_ltc_counter_metric", [](const std::string& name, int value) {
    lazy_tensors::metrics::Counter(name).AddValue(value);
  });
}

void InitMNMBindings(py::module m) {
  InitMNMModuleBindings(m);
}

}  // namespace

}  // namespace torch_lazy_tensors

PYBIND11_MODULE(_RAZORC, m) {
  try {
    torch::utils::cuda_lazy_init();
  } catch (const python_error&) {
    // Do nothing, CUDA not available.
  }
  torch_lazy_tensors::InitLtcBindings(m);
  torch_lazy_tensors::InitMNMBindings(m);
}
