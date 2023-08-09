/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/native_layer_norm_backward.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

NativeLayerNormBackward::NativeLayerNormBackward(const Value& grad_out, const Value& input, std::vector<int64_t> normalized_shape,
                                                const Value& mean, const Value& rstd, const Value& weight, const Value& bias
                                                )
    : Node(ir::OpKind(at::aten::native_layer_norm_backward),
           {grad_out, input, mean, rstd, weight, bias}, /*num_outputs=*/3, lazy_tensors::util::MHash(normalized_shape)), normalized_shape_(normalized_shape)
            {
  SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr NativeLayerNormBackward::Clone(OpList operands) const {
  return MakeNode<NativeLayerNormBackward>(operands.at(0), operands.at(1), normalized_shape_,
                                           operands.at(2), operands.at(3), operands.at(4), operands.at(5));
}

std::string NativeLayerNormBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << "normalized_shape= "<< normalized_shape_;
  return ss.str();
}
}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
