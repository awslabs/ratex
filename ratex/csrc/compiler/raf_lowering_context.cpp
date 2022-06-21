/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ratex/csrc/compiler/raf_lowering_context.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/lowering_context.h"
#include "lazy_tensors/computation_client/computation_client.h"
#include "client/base_computation_client.h"

#include "./utils.h"

#include "raf/ir.h"
#include "raf/value.h"
#include "raf/pass.h"
#include "raf/binding.h"

namespace torch_lazy_tensors {
namespace compiler {
namespace raf_backend {

using namespace raf::ir;
using namespace raf::value;
using namespace raf::pass;
using namespace raf::binding;
using raf::pass::extract_binding::ExtractBinding;

lazy_tensors::StatusOr<lazy_tensors::ProgramShape> GenericComputationRAF::GetProgramShape() const {
  Function func = Downcast<Function>(InferType(computation_));
  FuncType ty = Downcast<FuncType>(func->checked_type());
  std::vector<Shape> parameters;
  std::vector<std::string> parameter_names;
  Shape result = ToLTCShape(ty->ret_type);
  for (const auto& arg : ty->arg_types) {
    parameters.push_back(ToLTCShape(arg));
  }
  for (const auto& var : func->params) {
    parameter_names.push_back(var->name_hint());
  }
  return lazy_tensors::ProgramShape(parameters, parameter_names, result);
}

lazy_tensors::Shape RAFLoweringContext::GetResultShape(size_t index) const {
  Var root = GetResult(index);
  Expr body = InferType(ExtractBinding(root, GetParams()));
  return ToLTCShape(body->checked_type());
}

size_t RAFLoweringContext::AddResult(const ir::Output& output) {
  return AddResult(GetOutputOp(output));
}

size_t RAFLoweringContext::AddResult(const Var& op) {
  root_tuple_.push_back(op);
  return root_tuple_.size() - 1;
}

lazy_tensors::StatusOr<std::shared_ptr<lazy_tensors::GenericComputation>>
RAFLoweringContext::Build() {
  Tuple tup(Array<Expr>(root_tuple_.begin(), root_tuple_.end()));
  Var root = root_tuple_.size() > 1 ? BindSymbol(tup) : root_tuple_[0];
  std::vector<Var> params = GetParams();
  Expr body = ExtractBinding(root, params);
  Function func(Array<Var>(params.begin(), params.end()), body, {}, {});
  Array<Var> free_vars = FreeVars(func);
  LTC_CHECK(free_vars.size() == 0U);
  return std::shared_ptr<lazy_tensors::GenericComputation>(
      new GenericComputationRAF(func, model_states_, alias_));
}

std::vector<Var> RAFLoweringContext::GetParams() const {
  std::vector<Var> params;
  for (const auto& data : parameters_) {
    lazy_tensors::client::Data::OpaqueHandle handle = data->GetOpaqueHandle();
    params.push_back(parameters_map_.at(handle).param);
  }
  return params;
}

void RAFLoweringContext::LowerNodeToResult(const ir::Node* node) {
  AddResult(LowerNode(node));
}

Var RAFLoweringContext::GetOutputOp(const ir::Output& output) {
  auto it = emitted_outputs_.find(output);
  if (it == emitted_outputs_.end()) {
    auto post_order = ir::Util::ComputePostOrder(output.node, &emit_status_);
    for (auto node : post_order) {
      LowerNode(node);
    }
    LowerNode(output.node);
    // At this point the outpout better be present, otherwise there is an issue
    // with the lowering code.
    it = emitted_outputs_.find(output);
    LTC_CHECK(it != emitted_outputs_.end()) << "No RAF operation emitted for output: " << output;
  }
  return it->second;
}

Var RAFLoweringContext::LowerNode(const ir::Node* node) {
  Var result;
  result = LowerNodeToRAF(node, this);
  // TODO(@hzfan): catch lowering error and report
  if (node->num_outputs() > 1) {
    for (size_t i = 0; i < node->num_outputs(); ++i) {
      AssignOutputOp(ir::Output(node, i), BindSymbol(TupleGetItem(result, i)));
    }
  } else {
    AssignOutputOp(ir::Output(node, 0), result);
  }
  return result;
}

Var RAFLoweringContext::GetResult(size_t index) const {
  return root_tuple_.at(index);
}

void RAFLoweringContext::AssignOutputOp(const ir::Output& output, const raf::ir::Var& op) {
  emitted_outputs_[output] = op;
}

Var RAFLoweringContext::GetParameter(const std::shared_ptr<lazy_tensors::client::Data>& data) {
  lazy_tensors::client::Data::OpaqueHandle handle = data->GetOpaqueHandle();
  auto it = parameters_map_.find(handle);
  if (it == parameters_map_.end()) {
    std::vector<int64_t> shape;
    DType dtype;
    std::tie(shape, dtype) = ToRAFShape(data->shape());
    Array<PrimExpr> arr_shape;
    for (const auto& s : shape) {
      arr_shape.push_back(Integer(s));
    }
    TensorType tty(arr_shape, DataType(dtype.operator DLDataType()));
    Var param = MakeVar(absl::StrCat("p", parameters_.size()), tty);
    it = parameters_map_.emplace(handle, Parameter{param, parameters_.size()}).first;
    parameters_.push_back(data);
    if (static_cast<ratex::BaseComputationClient::BaseData*>(data.get())->is_param) {
      model_states_.insert(param);
    }
  }
  parameter_sequence_.push_back(it->second.index);
  return it->second.param;
}

void RAFLoweringContext::SetUpAlias(const lazy_tensors::ShapeIndex& output_index,
                                    int64_t param_number,
                                    const lazy_tensors::ShapeIndex& param_index) {
  // std::cout << "SetUpAlias" << std::endl;
  // std::cout << "output_index = " << std::endl;
  // for (size_t i = 0; i < output_index.size(); ++i) {
  //   std::cout << output_index[i] << ", ";
  // }
  // std::cout << std::endl;
  // std::cout << "param_number = " << param_number << std::endl;
  // std::cout << "param_index = " << std::endl;
  // for (size_t i = 0; i < param_index.size(); ++i) {
  //   std::cout << param_index[i] << ", ";
  // }
  // std::cout << std::endl;
  LTC_CHECK_EQ(output_index.size(), 1U);
  LTC_CHECK_EQ(param_index.size(), 0U);
  LTC_CHECK(alias_.find(param_number) == alias_.end());
  alias_[param_number] = output_index[0];
}

}  // namespace raf_backend
}  // namespace compiler

namespace ir {

std::unique_ptr<LoweringContext> LoweringContext::Create(
    const std::string& name, Device device, lazy_tensors::Span<const Node* const> post_order,
    Util::EmissionMap emit_status) {
  return std::make_unique<compiler::raf_backend::RAFLoweringContext>(name, device, post_order,
                                                                     emit_status);
}

std::unique_ptr<LoweringContext> LoweringContext::Create(const std::string& name, Device device) {
  return std::make_unique<compiler::raf_backend::RAFLoweringContext>(name, device);
}

}  // namespace ir
}  // namespace torch_lazy_tensors
