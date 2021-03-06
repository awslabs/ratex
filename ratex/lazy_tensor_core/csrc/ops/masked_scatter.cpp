/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/masked_scatter.h"

#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

MaskedScatter::MaskedScatter(const Value& input, const Value& mask, const Value& source)
    : Node(ir::OpKind(at::aten::masked_scatter), {input, mask, source}, input.shape(),
           /*num_outputs=*/1) {
}

NodePtr MaskedScatter::Clone(OpList operands) const {
  return MakeNode<MaskedScatter>(operands.at(0), operands.at(1), operands.at(2));
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
