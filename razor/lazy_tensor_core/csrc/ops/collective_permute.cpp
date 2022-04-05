/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/collective_permute.h"

#include "absl/strings/str_join.h"
#include "lazy_tensor_core/csrc/compiler/node_lowering.h"
#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/util.h"
#include "lazy_tensors/shape_util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

CollectivePermute::CollectivePermute(const Value& input, const Value& token,
                                     std::vector<std::pair<int64_t, int64_t>> source_target_pairs)
    : Node(ltc_collective_permute, {input, token},
           /*num_outputs=*/2, lazy_tensors::util::MHash(source_target_pairs)),
      source_target_pairs_(std::move(source_target_pairs)) {
  SetShapeDeferred([&]() { return compiler::NodeLowering::Get()->Infer(this); });
}

NodePtr CollectivePermute::Clone(OpList operands) const {
  return MakeNode<CollectivePermute>(operands.at(0), operands.at(1), source_target_pairs_);
}

std::string CollectivePermute::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", source_target_pairs=(";
  for (size_t i = 0; i < source_target_pairs_.size(); ++i) {
    ss << (i == 0 ? "(" : ", (");
    ss << source_target_pairs_[i].first << ", " << source_target_pairs_[i].second << ")";
  }
  ss << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
