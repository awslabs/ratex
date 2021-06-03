#include "torch/csrc/jit/python/pybind.h"
#include "lazy_tensor_core/csrc/aten_ltc_bridge.h"
#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/shape.h"
#include "lazy_tensor_core/csrc/ops/device_data.h"
#include "torch_mnm/csrc/ops/relay_expr.h"
#include "torch_mnm/csrc/ops/relay_function.h"
#include "torch_mnm/csrc/ops/mnm_ops.h"
#include "torch_mnm/csrc/compiler/utils.h"
#include "torch_mnm/csrc/mnm_model_state.h"
#include "mnm_client/mnm_computation_client.h"
#include "mnm/registry.h"
#include "meta/src/op/ty/utils.h"

namespace torch_lazy_tensors {

namespace {

using namespace ir;

// Value MakeTupleValue(ir::NodePtr node) {
//   std::vector<Value> ops;
//   ops.reserve(node->num_outputs());
//   for (size_t i = 0; i < node->num_outputs(); ++i) {
//     ops.emplace_back(node, i);
//   }
//   Value ret = ir::MakeNode<ir::ops::Tuple>(ops);
//   return ret;
// }

Value MarkParameter(Output out) {
  auto* device_data = ir::ops::DeviceData::Cast(out.node);
  if (device_data) {
    const auto* data = static_cast<const torch_mnm::MNMComputationClient::MNMData*>(device_data->data().get());
    NodePtr marked_device_data = std::make_shared<ir::ops::DeviceData>(
      std::make_shared<torch_mnm::MNMComputationClient::MNMData>(
        data->device(), data->shape(), data->handle, true));
    return marked_device_data;
  }
  std::vector<Value> new_ops;
  for (size_t i = 0; i < out.node->operands().size(); ++i) {
    Value new_op = MarkParameter(out.node->operands()[i]);
    new_ops.push_back(new_op);
  }
  NodePtr new_node = out.node->Clone(new_ops);
  return Value(new_node, out.index);
}

// torch_mnm::MNMComputationClient::MNMData* GetData(Output out) {
//   auto* device_data = ir::ops::DeviceData::Cast(out.node);
//   if (device_data) {
//     auto* data = static_cast<torch_mnm::MNMComputationClient::MNMData*>(device_data->data().get());
//     return data;
//   }
//   LTC_CHECK_EQ(out.node->operands().size(), 1U);
//   return GetData(out.node->operand(0));
// }

lazy_tensors::ComputationClient::DataPtr GetData(Output out) {
  auto* device_data = ir::ops::DeviceData::Cast(out.node);
  if (device_data) {
    return device_data->data();
  }
  for (size_t i = 0; i < out.node->operands().size(); ++i) {
    auto data = GetData(out.node->operand(i));
    if (data) {
      return data;
    }
  }
  return nullptr;
}

void InitMNMModuleBindings(py::module m) {
  m.def("_mnm_invoke_relay", [](at::Tensor func, const std::vector<at::Tensor>& tensors) -> std::vector<at::Tensor> {
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
    // bwd is closure, whose type cannot be expressed as at::ScalarType. Byte is used as dummy data type for it.
    Device dev(DeviceType::CPU, 0);
    // Approach 1: RelayExpr returns multiple nodes
    ir::NodePtr ret = ir::MakeNode<ir::ops::RelayExpr>(input_values);
    if (ret->shape().IsTuple()) {
      std::vector<at::Tensor> unpacked_ret;
      unpacked_ret.reserve(ret->shape().tuple_shapes_size());
      for (int i = 0; i < ret->shape().tuple_shapes_size(); ++i) {
        const at::ScalarType tuple_type = at::ScalarType::Byte;
        at::ScalarType scalar_type = ret->shape().tuple_shapes(i).IsTuple() ? tuple_type :
          TensorTypeFromLtcType(ret->shape().tuple_shapes(i).element_type());
        unpacked_ret.emplace_back(bridge::AtenFromLtcTensor(LazyTensor::Create(ir::Value(ret, i),
          dev, scalar_type)));
      }
      return unpacked_ret;
    } else {
      return {bridge::AtenFromLtcTensor(LazyTensor::Create(ir::Value(ret, 0), dev,
        TensorTypeFromLtcType(ret->shape().element_type())))};
    }
  });

  m.def("_mnm_to_tensor", [](int64_t handle) ->  at::Tensor {
    static auto handle_to_value = mnm::registry::GetPackedFunc("mnm.value.HandleToValue"); 
    // TODO(@hzfan): assign real data type when handle is TensorValue
    mnm::value::Value val = handle_to_value(handle);
    lazy_tensors::Shape shape = compiler::mnm_backend::ToLTCShape(tvm::relay::TupleType({mnm::op::GetType(val)}));
    LazyTensor ret;
    if (const auto* cvo = val.as<mnm::value::ClosureValueObj>()) {
      // ret is closure, whose type cannot be expressed as at::ScalarType. Byte is used as dummy data type for it. 
      LTC_CHECK_EQ(cvo->env.size(), 0U);
      Device dev(DeviceType::CPU, 0);
      ir::Value relay_function = ir::MakeNode<ir::ops::RelayFunction>(cvo->func);
      ret = LazyTensor::Create(relay_function, dev, at::ScalarType::Byte);
    } else {
      LTC_LOG(FATAL) << "Unsupported type " << val->GetTypeKey();
    }
    return bridge::AtenFromLtcTensor(ret);
  });

  m.def("_mnm_mark_parameter", [](at::Tensor tensor) ->  at::Tensor {
    LazyTensor lazy_tensor = bridge::GetLtcTensor(tensor);
    ir::Value ir_value = lazy_tensor.GetIrValue();
    GetMNMModelState()->AddModelState(lazy_tensor);
    // ir::Value marked_ir_value = MarkParameter(ir_value);
    // lazy_tensors::ComputationClient::DataPtr data = GetData(ir_value);
    // lazy_tensor.SetDataHandle(data);
    // auto mnm_data = static_cast<torch_mnm::MNMComputationClient::MNMData*>(data.get());
    // mnm_data->is_param = true;
    // std::cout << "ir_value: " << ir_value->ToString() << std::endl;
    return tensor;
    // return bridge::AtenFromLtcTensor(LazyTensor::Create(marked_ir_value, lazy_tensor.GetDevice(), lazy_tensor.dtype_optional())); 
    // auto* device_data = ir::ops::DeviceData::Cast(ir_value.operator->());
    // LTC_CHECK(device_data);
    // static_cast<const torch_mnm::MNMComputationClient::MNMData*>(device_data->data().get())->is_param = true;
// LazyTensor LazyTensor::Create(
//     ir::Value ir_value, const Device& device,
//     c10::optional<at::ScalarType> logical_element_type)
  });
}
  
void InitMNMBindings(py::module m) { InitMNMModuleBindings(m); }

}  // namespace

}  // namespace torch_lazy_tensors

PYBIND11_MODULE(_TORCHMNMC, m) {
  torch_lazy_tensors::InitMNMBindings(m);
}
