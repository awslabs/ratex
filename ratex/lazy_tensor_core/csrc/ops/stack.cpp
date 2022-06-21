/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/stack.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Stack::Stack(lazy_tensors::Span<const ir::Value> values, int64_t dim)
    : Node(ir::OpKind(at::aten::stack), values,
           /*num_outputs=*/1, lazy_tensors::util::MHash(dim)),
      dim_(dim) {
  SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr Stack::Clone(OpList operands) const {
  return MakeNode<Stack>(operands, dim_);
}

std::string Stack::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
