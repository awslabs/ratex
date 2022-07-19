/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/layer_norm.h"

#include "lazy_tensors/computation_client/debug_macros.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

LayerNorm::LayerNorm(const Value& input, std::vector<int64_t> normalized_shape, const Value& weight, const Value& bias, double eps, bool cudnn_enable)
    : Node(ir::OpKind(at::aten::layer_norm), {input, weight, bias}), normalized_shape_(normalized_shape), eps_(eps), cudnn_enable_(cudnn_enable){
    SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr LayerNorm::Clone(OpList operands) const {
  return MakeNode<LayerNorm>(operands.at(0), normalized_shape_, operands.at(1), operands.at(2), eps_, cudnn_enable_);
}

std::string LayerNorm::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << "normalized_shape= "<< normalized_shape_ << "eps= " << eps_ << "cudnn_enabled= " << cudnn_enable_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
