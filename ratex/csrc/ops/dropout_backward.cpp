/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ratex/csrc/ops/dropout_backward.h"
#include "ratex/csrc/ops/raf_ops.h"

#include "absl/strings/str_join.h"
#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensor_core/csrc/torch_util.h"
#include "lazy_tensors/computation_client/util.h"

#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

DropoutBackward::DropoutBackward(const Value& grad_output, const Value& mask,
                                 const Value& reserve_space)
    : Node(raf_dropout_backward, {grad_output, mask, reserve_space},
           /*num_outputs=*/1) {
  SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr DropoutBackward::Clone(OpList operands) const {
  return MakeNode<DropoutBackward>(operands.at(0), operands.at(1), operands.at(2));
}

std::string DropoutBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString();
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
