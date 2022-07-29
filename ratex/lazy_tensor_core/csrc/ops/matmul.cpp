/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/matmul.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

MatMul::MatMul(const Value& input, const Value& other, std::vector<int64_t> a_shape,
               std::vector<int64_t> b_shape)
    : Node(ir::OpKind(at::aten::matmul), {input, other}),
      a_shape_(a_shape),
      b_shape_(b_shape) {
  SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr MatMul::Clone(OpList operands) const {
  return MakeNode<MatMul>(operands.at(0), operands.at(1), a_shape_, b_shape_);
}

std::string MatMul::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << " a_shape= " << a_shape_ << " b_shape " << b_shape_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
