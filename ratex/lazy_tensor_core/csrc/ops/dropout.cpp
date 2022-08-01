/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/dropout.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Dropout::Dropout(const Value& input, double p)
    : Node(ir::OpKind(at::aten::dropout), {input}, 3, lazy_tensors::util::MHash(p)), p_(p) {
  SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr Dropout::Clone(OpList operands) const {
  return MakeNode<Dropout>(operands.at(0), p_);
}

std::string Dropout::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << " p=" << p_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
