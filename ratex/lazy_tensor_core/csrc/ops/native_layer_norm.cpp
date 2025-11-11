/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/native_layer_norm.h"

#include "lazy_tensors/computation_client/debug_macros.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {
//NativeLayerNorm Node
NativeLayerNorm::NativeLayerNorm(const Value& input, std::vector<int64_t> normalized_shape, const Value& weight, const Value& bias, double eps)
    : Node(ir::OpKind(at::aten::layer_norm), {input, weight, bias}, 3), normalized_shape_(normalized_shape), eps_(eps){
    SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr NativeLayerNorm::Clone(OpList operands) const {
  return MakeNode<NativeLayerNorm>(operands.at(0), normalized_shape_, operands.at(1), operands.at(2), eps_);
}

std::string NativeLayerNorm::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << "normalized_shape= "<< normalized_shape_ << "eps= " << eps_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
