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
#include "client/mnm_computation_client.h"
#include "mnm/registry.h"
#include "meta/src/op/ty/utils.h"

namespace torch_lazy_tensors {

namespace {

using namespace ir;

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
                LazyTensor::Create(ir::Value(ret, i), dev, scalar_type)
                    .ShallowCopyTo(&lazy_tensors[inplace_update_out_2_arg_idxs.at(i)]);
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

  m.def("_mnm_mark_parameter", [](at::Tensor tensor) -> at::Tensor {
    LazyTensor lazy_tensor = bridge::GetLtcTensor(tensor);
    ir::Value ir_value = lazy_tensor.GetIrValue();
    GetMNMModelState()->AddModelState(lazy_tensor);
    return tensor;
  });
}

void InitMNMBindings(py::module m) {
  InitMNMModuleBindings(m);
}

}  // namespace

}  // namespace torch_lazy_tensors

PYBIND11_MODULE(_TORCHMNMC, m) {
  torch_lazy_tensors::InitMNMBindings(m);
}
