/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/unselect.h"

#include "lazy_tensor_core/csrc/ops/ltc_ops.h"
#include "lazy_tensor_core/csrc/ops/select.h"
#include "lazy_tensor_core/csrc/tensor_util.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Unselect::Unselect(const Value& target, const Value& source, int64_t dim, int64_t start,
                   int64_t end, int64_t stride)
    : Node(ltc_unselect, {target, source}, target.shape(),
           /*num_outputs=*/1, lazy_tensors::util::MHash(dim, start, end, stride)),
      dim_(dim),
      start_(start),
      end_(end),
      stride_(stride) {
}

NodePtr Unselect::Clone(OpList operands) const {
  return MakeNode<Unselect>(operands.at(0), operands.at(1), dim_, start_, end_, stride_);
}

std::string Unselect::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_ << ", start=" << start_ << ", end=" << end_
     << ", stride=" << stride_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
