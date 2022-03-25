/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "lazy_tensor_core/csrc/ops/all_gather.h"

#include "absl/strings/str_join.h"
#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

AllGather::AllGather(const Value& input, const Value& token, int64_t dim, int64_t shard_count,
                     std::vector<std::vector<int64_t>> groups)
    : Node(ltc_all_gather, {input, token},
           /*num_outputs=*/2, lazy_tensors::util::MHash(dim, shard_count, groups)),
      dim_(dim),
      shard_count_(shard_count),
      groups_(std::move(groups)) {
  SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr AllGather::Clone(OpList operands) const {
  return MakeNode<AllGather>(operands.at(0), operands.at(1), dim_, shard_count_, groups_);
}

std::string AllGather::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_ << ", shard_count=" << shard_count_ << ", groups=(";
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
