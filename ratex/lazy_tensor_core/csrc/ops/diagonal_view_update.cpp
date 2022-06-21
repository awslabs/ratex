/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/diagonal_view_update.h"

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

DiagonalViewUpdate::DiagonalViewUpdate(const Value& target, const Value& input, int64_t offset,
                                       int64_t dim1, int64_t dim2)
    : Node(ltc_diagonal_view_update, {target, input}, target.shape(),
           /*num_outputs=*/1, lazy_tensors::util::MHash(offset, dim1, dim2)),
      offset_(offset),
      dim1_(dim1),
      dim2_(dim2) {
}

NodePtr DiagonalViewUpdate::Clone(OpList operands) const {
  return MakeNode<DiagonalViewUpdate>(operands.at(0), operands.at(1), offset_, dim1_, dim2_);
}

std::string DiagonalViewUpdate::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", offset=" << offset_ << ", dim1=" << dim1_ << ", dim2=" << dim2_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
