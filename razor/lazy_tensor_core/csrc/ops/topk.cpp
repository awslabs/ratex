/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/topk.h"

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

TopK::TopK(const Value& input, int64_t k, int64_t dim, bool largest, bool sorted)
    : Node(ir::OpKind(at::aten::topk), {input},
           /*num_outputs=*/2, lazy_tensors::util::MHash(k, dim, largest, sorted)),
      k_(k),
      dim_(dim),
      largest_(largest),
      sorted_(sorted) {
  SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr TopK::Clone(OpList operands) const {
  return MakeNode<TopK>(operands.at(0), k_, dim_, largest_, sorted_);
}

std::string TopK::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", k=" << k_ << ", dim=" << dim_ << ", largest=" << largest_
     << ", sorted=" << sorted_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
