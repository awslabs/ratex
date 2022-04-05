/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/tril.h"

#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Tril::Tril(const Value& input, int64_t diagonal)
    : Node(ir::OpKind(at::aten::tril), {input}, input.shape(),
           /*num_outputs=*/1, lazy_tensors::util::MHash(diagonal)),
      diagonal_(diagonal) {
}

NodePtr Tril::Clone(OpList operands) const {
  return MakeNode<Tril>(operands.at(0), diagonal_);
}

std::string Tril::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", diagonal=" << diagonal_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
