/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modifications Copyright (c) Facebook, Inc.
 */

#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// This node has no metadata, so it could have been implemented as generic-op in
// ops.cpp, but since this might require special handling from upper IR layers,
// it gets its own IR node class.
class NonZero : public Node {
 public:
  NonZero(const Value& input);

  NodePtr Clone(OpList operands) const override;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors