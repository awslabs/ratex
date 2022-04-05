/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/index_put.h"

#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

IndexPut::IndexPut(const ir::Value& base, const ir::Value& indices, int64_t start_dim,
                   const ir::Value& values, bool accumulate)
    : Node(OpKind(at::aten::index_put), {base, indices, values}, base.shape(),
           /*num_outputs=*/1, lazy_tensors::util::MHash(start_dim, accumulate)),
      start_dim_(start_dim),
      accumulate_(accumulate) {
}

std::string IndexPut::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", start_dim=" << start_dim_ << ", accumulate=" << accumulate_;
  return ss.str();
}

NodePtr IndexPut::Clone(OpList operands) const {
  return MakeNode<IndexPut>(operands.at(0), operands.at(1), start_dim_, operands.at(2),
                            accumulate_);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
