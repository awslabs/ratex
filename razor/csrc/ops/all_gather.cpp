/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "razor/csrc/ops/all_gather.h"

#include "absl/strings/str_join.h"
#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

#include "razor/csrc/ops/mnm_ops.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

MNMAllGather::MNMAllGather(lazy_tensors::Span<const Value> operands, lazy_tensors::int64 dim,
                           std::vector<std::vector<lazy_tensors::int64>> groups)
    : Node(mnm_all_gather, operands, operands.size(), lazy_tensors::util::MHash(dim, groups)),
      dim_(dim),
      groups_(std::move(groups)) {
  SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr MNMAllGather::Clone(OpList operands) const {
  return MakeNode<MNMAllGather>(operands, dim_, groups_);
}

std::string MNMAllGather::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_ << ", groups=(";
  for (size_t i = 0; i < groups_.size(); ++i) {
    ss << (i == 0 ? "(" : ",(");
    ss << absl::StrJoin(groups_[i], ", ") << ")";
  }
  ss << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
