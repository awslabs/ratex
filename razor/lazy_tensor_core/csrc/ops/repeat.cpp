/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/repeat.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/debug_macros.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Repeat::Repeat(const Value& input, std::vector<int64_t> repeats)
    : Node(ir::OpKind(at::aten::repeat), {input},
           /*num_outputs=*/1, lazy_tensors::util::MHash(repeats)),
      repeats_(std::move(repeats)) {
  SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr Repeat::Clone(OpList operands) const {
  return MakeNode<Repeat>(operands.at(0), repeats_);
}

std::string Repeat::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", repeats=(" << absl::StrJoin(repeats_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
