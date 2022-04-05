/*
 * Copyright (c) 2018 Google Inc. All Rights Reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lazy_tensor_core/csrc/ops/exponential.h"

#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

Exponential::Exponential(const Value& lambda, const Value& seed, lazy_tensors::Shape shape)
    : Node(ir::OpKind(at::aten::exponential), {lambda, seed}, std::move(shape)) {
}

NodePtr Exponential::Clone(OpList operands) const {
  return MakeNode<Exponential>(operands.at(0), operands.at(1), shape());
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
